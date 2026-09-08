from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from inference.config import Settings
from inference.service import create_app


@pytest.mark.asyncio
async def test_prediction_readiness_runtime_and_metrics() -> None:
    app = create_app(Settings(max_batch_size=8, max_wait_ms=10, max_queue_size=32))
    async with app.router.lifespan_context(app):  # noqa: SIM117 - explicit lifecycle under test.
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            ready = await client.get("/ready")
            assert ready.status_code == 200
            assert ready.json()["runtime"]["queue_capacity"] == 32

            responses = await asyncio.gather(
                *(
                    client.post(
                        "/predict",
                        json={"features": [float(i) / 10, 0.2, 0.5, -0.1]},
                    )
                    for i in range(12)
                )
            )
            assert all(response.status_code == 200 for response in responses)
            assert {response.json()["version"] for response in responses} == {"2026.08"}

            runtime = (await client.get("/runtime")).json()
            assert runtime["batcher"]["completed"] == 12
            metrics = await client.get("/metrics")
            assert 'inference_requests_total{status="success"} 12.0' in metrics.text


@pytest.mark.asyncio
async def test_admin_reload_is_disabled_by_default() -> None:
    app = create_app(Settings())
    async with app.router.lifespan_context(app):  # noqa: SIM117 - explicit lifecycle under test.
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/admin/models/reload",
                json={"weights": [1, 1, 1, 1], "bias": 0, "name": "next", "version": "2"},
            )
            assert response.status_code == 404


@pytest.mark.asyncio
async def test_authenticated_reload_changes_subsequent_predictions() -> None:
    app = create_app(Settings(admin_token="secret"))
    async with app.router.lifespan_context(app):  # noqa: SIM117 - explicit lifecycle under test.
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            denied = await client.post(
                "/admin/models/reload",
                json={"weights": [1, 1, 1, 1], "bias": 0, "name": "next", "version": "2"},
            )
            assert denied.status_code == 403
            accepted = await client.post(
                "/admin/models/reload",
                headers={"X-Admin-Token": "secret"},
                json={"weights": [1, 1, 1, 1], "bias": 0, "name": "next", "version": "2"},
            )
            assert accepted.status_code == 200
            assert accepted.json()["model"]["generation"] == 2
            prediction = await client.post("/predict", json={"features": [0, 0, 0, 0]})
            assert prediction.json()["version"] == "2"


@pytest.mark.asyncio
async def test_readiness_is_distinct_from_liveness_before_startup() -> None:
    app = create_app(Settings())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/ready")).status_code == 503
