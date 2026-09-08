from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from inference.config import Settings
from inference.service import create_app
from load_test import percentile, run_load


def test_percentile_interpolates_and_handles_empty_input() -> None:
    assert percentile([], 0.95) == 0
    assert percentile([10], 0.95) == 10
    assert percentile([0, 10], 0.5) == 5


@pytest.mark.asyncio
async def test_load_report_is_machine_readable_and_slo_gated() -> None:
    app = create_app(Settings(max_batch_size=8, max_wait_ms=1))
    async with app.router.lifespan_context(app):  # noqa: SIM117 - explicit lifecycle under test.
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            report = await run_load(
                client,
                url="/predict",
                requests=30,
                concurrency=6,
                max_p95_ms=1_000,
                min_success_rate=1.0,
            )
    assert report.succeeded == 30
    assert report.status_counts == {"200": 30}
    assert report.slo_passed
    assert report.throughput_rps > 0


@pytest.mark.asyncio
async def test_load_report_explains_failed_latency_budget() -> None:
    app = create_app(Settings())
    async with app.router.lifespan_context(app):  # noqa: SIM117 - explicit lifecycle under test.
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            report = await run_load(
                client,
                url="/predict",
                requests=2,
                concurrency=1,
                max_p95_ms=0,
                min_success_rate=1.0,
            )
    assert not report.slo_passed
    assert "exceeds" in report.slo_failures[0]
