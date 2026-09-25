from __future__ import annotations

import asyncio
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from inference.config import Settings
from inference.open_loop import (
    LoadConfigurationError,
    LoadTransportError,
    OpenLoopPolicy,
    parse_args,
    percentile,
    run_open_loop,
    validate_target_url,
)
from inference.service import create_app


class Response:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code


class VirtualTime:
    def __init__(self, *, overshoot: float = 0.0) -> None:
        self.value = 100.0
        self.overshoot = overshoot

    def clock(self) -> float:
        return self.value

    async def sleep(self, delay: float) -> None:
        await asyncio.sleep(0)
        self.value += delay + (self.overshoot if delay > 0 else 0.0)


class VirtualClient:
    def __init__(self, virtual_time: VirtualTime, *, status_code: int = 200) -> None:
        self.virtual_time = virtual_time
        self.status_code = status_code
        self.calls = 0

    async def post(self, url: str, *, json: object) -> Response:
        assert url == "https://inference.example/predict"
        assert json == {"features": [self.calls % 10 / 10.0, 0.2, 0.5, -0.1]}
        self.calls += 1
        self.virtual_time.value += 0.001
        return Response(self.status_code)


def policy(**changes: object) -> OpenLoopPolicy:
    return replace(
        OpenLoopPolicy(
            requests=5,
            rate_rps=100.0,
            max_in_flight=1,
            max_scheduled_p95_ms=100.0,
            max_scheduler_lag_p95_ms=100.0,
            min_success_rate=1.0,
            max_client_drop_rate=0.0,
        ),
        **changes,
    )


def test_percentile_interpolates_and_rejects_invalid_quantile() -> None:
    assert percentile([], 0.95) == 0.0
    assert percentile([10.0], 0.95) == 10.0
    assert percentile([0.0, 10.0], 0.5) == 5.0
    with pytest.raises(ValueError, match="quantile"):
        percentile([1.0], 1.1)


@pytest.mark.parametrize(
    "candidate",
    [
        OpenLoopPolicy(requests=True),
        OpenLoopPolicy(requests=0),
        OpenLoopPolicy(rate_rps=math.nan),
        OpenLoopPolicy(rate_rps=math.inf),
        OpenLoopPolicy(max_in_flight=0),
        OpenLoopPolicy(min_success_rate=1.1),
        OpenLoopPolicy(max_client_drop_rate=-0.1),
    ],
)
def test_policy_rejects_unsafe_or_unbounded_values(candidate: OpenLoopPolicy) -> None:
    with pytest.raises(LoadConfigurationError):
        candidate.validated()


@pytest.mark.parametrize(
    "url",
    [
        "",
        "inference.example/predict",
        "ftp://inference.example/predict",
        "https://user:secret@inference.example/predict",
        "https://inference.example/predict?token=secret",
        "https://inference.example/predict#fragment",
    ],
)
def test_target_url_rejects_ambiguous_or_secret_bearing_values(url: str) -> None:
    with pytest.raises(LoadConfigurationError):
        validate_target_url(url)


@pytest.mark.asyncio
async def test_fixed_rate_profile_reports_service_and_scheduled_latency_separately() -> None:
    virtual_time = VirtualTime()
    client = VirtualClient(virtual_time)
    report = await run_open_loop(
        client,
        url="https://inference.example/predict",
        policy=policy(),
        clock=virtual_time.clock,
        sleep=virtual_time.sleep,
    )

    assert client.calls == 5
    assert report.requested == report.admitted == report.succeeded == 5
    assert report.client_dropped == 0
    assert report.status_counts == {"200": 5}
    assert report.service_latency_p95_ms == pytest.approx(1.0)
    assert report.scheduled_latency_p95_ms >= report.service_latency_p95_ms
    assert report.schedule_span_seconds == pytest.approx(0.04)
    assert report.slo_passed is True
    assert len(report.evidence_sha256) == 64
    json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.asyncio
async def test_report_is_deterministic_for_identical_clock_and_response_evidence() -> None:
    first_time = VirtualTime()
    second_time = VirtualTime()
    first = await run_open_loop(
        VirtualClient(first_time),
        url="https://inference.example/predict",
        policy=policy(),
        clock=first_time.clock,
        sleep=first_time.sleep,
    )
    second = await run_open_loop(
        VirtualClient(second_time),
        url="https://inference.example/predict",
        policy=policy(),
        clock=second_time.clock,
        sleep=second_time.sleep,
    )
    assert first == second


@pytest.mark.asyncio
async def test_scheduler_overshoot_is_included_in_scheduled_latency_and_gate() -> None:
    virtual_time = VirtualTime(overshoot=0.005)
    report = await run_open_loop(
        VirtualClient(virtual_time),
        url="https://inference.example/predict",
        policy=policy(max_scheduler_lag_p95_ms=4.0),
        clock=virtual_time.clock,
        sleep=virtual_time.sleep,
    )
    assert report.service_latency_p95_ms == pytest.approx(1.0)
    assert report.scheduler_lag_p95_ms >= 5.0
    assert report.scheduled_latency_p95_ms > report.service_latency_p95_ms
    assert report.slo_passed is False
    assert "scheduler lag p95" in report.slo_failures[0]


@pytest.mark.asyncio
async def test_pending_work_is_bounded_and_excess_arrivals_are_counted_as_drops() -> None:
    class SlowClient:
        def __init__(self) -> None:
            self.calls = 0

        async def post(self, url: str, *, json: object) -> Response:
            self.calls += 1
            await asyncio.sleep(0.1)
            return Response()

    client = SlowClient()
    report = await run_open_loop(
        client,
        url="http://127.0.0.1:8000/predict",
        policy=policy(requests=6, rate_rps=10_000.0, max_client_drop_rate=0.0),
    )
    assert client.calls == report.admitted == 1
    assert report.client_dropped == 5
    assert report.status_counts == {"200": 1, "client_overload": 5}
    assert report.success_rate == pytest.approx(1 / 6)
    assert report.client_drop_rate == pytest.approx(5 / 6)
    assert report.slo_passed is False
    assert any("client drop rate" in failure for failure in report.slo_failures)


@pytest.mark.asyncio
async def test_transport_errors_are_counted_without_exposing_exception_text() -> None:
    class FailingClient:
        async def post(self, url: str, *, json: object) -> Response:
            virtual_time.value += 0.001
            raise LoadTransportError("secret upstream address")

    virtual_time = VirtualTime()
    report = await run_open_loop(
        FailingClient(),
        url="https://inference.example/predict",
        policy=policy(requests=1),
        clock=virtual_time.clock,
        sleep=virtual_time.sleep,
    )
    assert report.status_counts == {"transport_error": 1}
    assert report.succeeded == 0
    assert "secret" not in json.dumps(report.to_dict())


@pytest.mark.asyncio
async def test_non_200_response_counts_as_failed_request() -> None:
    virtual_time = VirtualTime()
    report = await run_open_loop(
        VirtualClient(virtual_time, status_code=503),
        url="https://inference.example/predict",
        policy=policy(requests=1),
        clock=virtual_time.clock,
        sleep=virtual_time.sleep,
    )
    assert report.status_counts == {"503": 1}
    assert report.failed == 1
    assert report.slo_passed is False


@pytest.mark.asyncio
async def test_open_loop_profile_exercises_real_inference_asgi_path() -> None:
    app = create_app(Settings(max_batch_size=4, max_wait_ms=1, max_queue_size=16))
    async with app.router.lifespan_context(app):  # noqa: SIM117 - lifecycle owns client scope.
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://inference.test"
        ) as client:
            report = await run_open_loop(
                client,
                url="http://inference.test/predict",
                policy=policy(
                    requests=10,
                    rate_rps=100.0,
                    max_in_flight=4,
                    max_scheduled_p95_ms=1_000.0,
                    max_scheduler_lag_p95_ms=1_000.0,
                ),
            )
    assert report.succeeded == 10
    assert report.client_dropped == 0
    assert report.status_counts == {"200": 10}
    assert report.slo_passed is True


@pytest.mark.asyncio
async def test_report_write_is_atomic_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    virtual_time = VirtualTime()
    report = await run_open_loop(
        VirtualClient(virtual_time),
        url="https://inference.example/predict",
        policy=policy(requests=1),
        clock=virtual_time.clock,
        sleep=virtual_time.sleep,
    )
    output = tmp_path / "evidence.json"
    report.write_json(output)
    assert json.loads(output.read_text())["evidence_sha256"] == report.evidence_sha256
    assert not (tmp_path / "evidence.json.tmp").exists()


def test_cli_rejects_non_finite_policy_before_network_access(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inference.open_loop",
            "--rate-rps",
            "nan",
            "--output",
            str(tmp_path / "should-not-exist.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert json.loads(result.stdout)["error"] == "INVALID_LOAD_CONFIGURATION"
    assert not (tmp_path / "should-not-exist.json").exists()


def test_proxy_environment_is_opt_in() -> None:
    assert parse_args([]).trust_env is False
    assert parse_args(["--trust-env"]).trust_env is True
