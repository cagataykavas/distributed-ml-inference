"""Bounded open-loop HTTP load profile for inference endpoints."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


class LoadConfigurationError(ValueError):
    """The load profile cannot be executed safely."""


class LoadTransportError(RuntimeError):
    """A transport adapter could not complete one request."""


class LoadArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise LoadConfigurationError(f"invalid command line: {message}")


class ResponseLike(Protocol):
    status_code: int


class AsyncPostClient(Protocol):
    async def post(self, url: str, *, json: object) -> ResponseLike: ...


@dataclass(frozen=True, slots=True)
class OpenLoopPolicy:
    requests: int = 500
    rate_rps: float = 100.0
    max_in_flight: int = 50
    max_scheduled_p95_ms: float = 250.0
    max_scheduler_lag_p95_ms: float = 25.0
    min_success_rate: float = 0.99
    max_client_drop_rate: float = 0.01

    def validated(self) -> OpenLoopPolicy:
        _strict_int(self.requests, "requests", 1, 100_000)
        _strict_int(self.max_in_flight, "max_in_flight", 1, 10_000)
        _finite_float(self.rate_rps, "rate_rps", 0.001, 1_000_000.0)
        _finite_float(
            self.max_scheduled_p95_ms,
            "max_scheduled_p95_ms",
            0.0,
            86_400_000.0,
        )
        _finite_float(
            self.max_scheduler_lag_p95_ms,
            "max_scheduler_lag_p95_ms",
            0.0,
            86_400_000.0,
        )
        _finite_float(self.min_success_rate, "min_success_rate", 0.0, 1.0)
        _finite_float(self.max_client_drop_rate, "max_client_drop_rate", 0.0, 1.0)
        return self


@dataclass(frozen=True, slots=True)
class OpenLoopReport:
    schema_version: int
    mode: str
    requested: int
    admitted: int
    succeeded: int
    failed: int
    client_dropped: int
    max_in_flight: int
    target_rate_rps: float
    schedule_span_seconds: float
    elapsed_seconds: float
    completion_throughput_rps: float
    success_rate: float
    client_drop_rate: float
    service_latency_p50_ms: float
    service_latency_p95_ms: float
    service_latency_p99_ms: float
    service_latency_max_ms: float
    scheduled_latency_p50_ms: float
    scheduled_latency_p95_ms: float
    scheduled_latency_p99_ms: float
    scheduled_latency_max_ms: float
    scheduler_lag_p50_ms: float
    scheduler_lag_p95_ms: float
    scheduler_lag_p99_ms: float
    scheduler_lag_max_ms: float
    status_counts: dict[str, int]
    slo_passed: bool
    slo_failures: tuple[str, ...]
    evidence_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)


def _strict_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LoadConfigurationError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise LoadConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


def _finite_float(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LoadConfigurationError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise LoadConfigurationError(f"{name} must be finite and between {minimum} and {maximum}")
    return parsed


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile without third-party dependencies."""
    if not values:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_summary(values: Sequence[float]) -> tuple[float, float, float, float]:
    return (
        percentile(values, 0.50),
        percentile(values, 0.95),
        percentile(values, 0.99),
        max(values, default=0.0),
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_target_url(url: str) -> str:
    if not isinstance(url, str) or len(url) > 2_048:
        raise LoadConfigurationError("url must be a string no longer than 2048 characters")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise LoadConfigurationError("url must be an absolute HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise LoadConfigurationError("url credentials are not allowed")
    if parsed.query:
        raise LoadConfigurationError("url query parameters are not allowed")
    if parsed.fragment:
        raise LoadConfigurationError("url fragments are not allowed")
    return url


async def run_open_loop(
    client: AsyncPostClient,
    *,
    url: str,
    policy: OpenLoopPolicy | None = None,
    clock: Clock = time.perf_counter,
    sleep: Sleep = asyncio.sleep,
) -> OpenLoopReport:
    """Offer requests at fixed scheduled times without creating an unbounded waiter queue."""
    active_policy = (policy or OpenLoopPolicy()).validated()
    target_url = validate_target_url(url)
    statuses: Counter[str] = Counter()
    service_latencies: list[float] = []
    scheduled_latencies: list[float] = []
    scheduler_lags: list[float] = []
    pending: set[asyncio.Task[None]] = set()
    started_at = clock()
    if not math.isfinite(started_at):
        raise LoadConfigurationError("clock must return a finite value")

    def duration_ms(later: float, earlier: float) -> float:
        difference = later - earlier
        if not math.isfinite(later) or difference < 0:
            raise LoadConfigurationError("clock must be finite and monotonic")
        return difference * 1_000.0

    async def execute(index: int, scheduled_at: float) -> None:
        dispatched_at = clock()
        scheduler_lags.append(duration_ms(dispatched_at, scheduled_at))
        try:
            response = await client.post(
                target_url,
                json={"features": [index % 10 / 10.0, 0.2, 0.5, -0.1]},
            )
            status_code = response.status_code
            if isinstance(status_code, bool) or not isinstance(status_code, int):
                raise LoadConfigurationError("client response status_code must be an integer")
            if not 100 <= status_code <= 599:
                raise LoadConfigurationError("client response status_code is outside HTTP range")
            statuses[str(status_code)] += 1
        except LoadTransportError:
            statuses["transport_error"] += 1
        finally:
            completed_at = clock()
            service_latencies.append(duration_ms(completed_at, dispatched_at))
            scheduled_latencies.append(duration_ms(completed_at, scheduled_at))

    for index in range(active_policy.requests):
        scheduled_at = started_at + index / active_policy.rate_rps
        delay = max(0.0, scheduled_at - clock())
        await sleep(delay)
        completed = {task for task in pending if task.done()}
        if completed:
            await asyncio.gather(*completed)
            pending.difference_update(completed)

        if len(pending) >= active_policy.max_in_flight:
            scheduler_lags.append(duration_ms(clock(), scheduled_at))
            statuses["client_overload"] += 1
            continue
        pending.add(asyncio.create_task(execute(index, scheduled_at)))

    if pending:
        await asyncio.gather(*pending)
    completed_at = clock()
    elapsed_seconds = completed_at - started_at
    if not math.isfinite(completed_at) or elapsed_seconds <= 0:
        raise LoadConfigurationError("clock must advance during the load profile")

    client_dropped = statuses["client_overload"]
    admitted = active_policy.requests - client_dropped
    succeeded = statuses["200"]
    success_rate = succeeded / active_policy.requests
    client_drop_rate = client_dropped / active_policy.requests
    service = _metric_summary(service_latencies)
    scheduled = _metric_summary(scheduled_latencies)
    scheduler = _metric_summary(scheduler_lags)
    failures: list[str] = []
    if scheduled[1] > active_policy.max_scheduled_p95_ms:
        failures.append(
            f"scheduled p95 {scheduled[1]:.2f}ms exceeds {active_policy.max_scheduled_p95_ms:.2f}ms"
        )
    if scheduler[1] > active_policy.max_scheduler_lag_p95_ms:
        failures.append(
            f"scheduler lag p95 {scheduler[1]:.2f}ms exceeds "
            f"{active_policy.max_scheduler_lag_p95_ms:.2f}ms"
        )
    if success_rate < active_policy.min_success_rate:
        failures.append(
            f"success rate {success_rate:.4f} is below {active_policy.min_success_rate:.4f}"
        )
    if client_drop_rate > active_policy.max_client_drop_rate:
        failures.append(
            f"client drop rate {client_drop_rate:.4f} exceeds "
            f"{active_policy.max_client_drop_rate:.4f}"
        )

    report_data: dict[str, object] = {
        "schema_version": 1,
        "mode": "open_loop",
        "requested": active_policy.requests,
        "admitted": admitted,
        "succeeded": succeeded,
        "failed": active_policy.requests - succeeded,
        "client_dropped": client_dropped,
        "max_in_flight": active_policy.max_in_flight,
        "target_rate_rps": active_policy.rate_rps,
        "schedule_span_seconds": (active_policy.requests - 1) / active_policy.rate_rps,
        "elapsed_seconds": elapsed_seconds,
        "completion_throughput_rps": succeeded / elapsed_seconds,
        "success_rate": success_rate,
        "client_drop_rate": client_drop_rate,
        "service_latency_p50_ms": service[0],
        "service_latency_p95_ms": service[1],
        "service_latency_p99_ms": service[2],
        "service_latency_max_ms": service[3],
        "scheduled_latency_p50_ms": scheduled[0],
        "scheduled_latency_p95_ms": scheduled[1],
        "scheduled_latency_p99_ms": scheduled[2],
        "scheduled_latency_max_ms": scheduled[3],
        "scheduler_lag_p50_ms": scheduler[0],
        "scheduler_lag_p95_ms": scheduler[1],
        "scheduler_lag_p99_ms": scheduler[2],
        "scheduler_lag_max_ms": scheduler[3],
        "status_counts": dict(sorted(statuses.items())),
        "slo_passed": not failures,
        "slo_failures": tuple(failures),
    }
    return OpenLoopReport(**report_data, evidence_sha256=_canonical_sha256(report_data))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = LoadArgumentParser(description="Run a bounded open-loop inference load profile")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--rate-rps", type=float, default=100.0)
    parser.add_argument("--max-in-flight", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-scheduled-p95-ms", type=float, default=250.0)
    parser.add_argument("--max-scheduler-lag-p95-ms", type=float, default=25.0)
    parser.add_argument("--min-success-rate", type=float, default=0.99)
    parser.add_argument("--max-client-drop-rate", type=float, default=0.01)
    parser.add_argument(
        "--trust-env",
        action="store_true",
        help="allow HTTP_PROXY/HTTPS_PROXY and other HTTP client environment settings",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/open-loop-load-report.json"))
    return parser.parse_args(argv)


async def run_from_args(args: argparse.Namespace) -> OpenLoopReport:
    timeout = _finite_float(args.timeout_seconds, "timeout_seconds", 0.001, 86_400.0)
    policy = OpenLoopPolicy(
        requests=args.requests,
        rate_rps=args.rate_rps,
        max_in_flight=args.max_in_flight,
        max_scheduled_p95_ms=args.max_scheduled_p95_ms,
        max_scheduler_lag_p95_ms=args.max_scheduler_lag_p95_ms,
        min_success_rate=args.min_success_rate,
        max_client_drop_rate=args.max_client_drop_rate,
    ).validated()
    url = f"{args.base_url.rstrip('/')}/predict"
    try:
        import httpx
    except ImportError as exc:
        raise LoadConfigurationError(
            "the CLI requires the dev extra: pip install 'distributed-ml-inference[dev]'"
        ) from exc

    class HttpxTransport:
        def __init__(self, client: httpx.AsyncClient) -> None:
            self.client = client

        async def post(self, target: str, *, json: object) -> ResponseLike:
            try:
                return await self.client.post(target, json=json)
            except httpx.HTTPError as exc:
                raise LoadTransportError from exc

    try:
        limits = httpx.Limits(
            max_connections=policy.max_in_flight,
            max_keepalive_connections=policy.max_in_flight,
        )
        client = httpx.AsyncClient(
            timeout=timeout,
            trust_env=args.trust_env,
            limits=limits,
        )
    except ImportError as exc:
        raise LoadConfigurationError("HTTP client transport initialization failed") from exc
    async with client:
        return await run_open_loop(HttpxTransport(client), url=url, policy=policy)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        report = asyncio.run(run_from_args(args))
        report.write_json(args.output)
    except LoadConfigurationError as exc:
        print(json.dumps({"error": "INVALID_LOAD_CONFIGURATION", "message": str(exc)}))
        return 3
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.slo_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
