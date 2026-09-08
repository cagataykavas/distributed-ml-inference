from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from batcher import BatchContractError, DynamicBatcher
from inference.model import LinearRiskModel
from inference.service import app


async def stop_worker(worker: asyncio.Task) -> None:
    worker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await worker


@pytest.mark.asyncio
async def test_dynamic_batcher_preserves_request_order() -> None:
    observed_batch_sizes: list[int] = []

    def predict(rows: list[int]) -> list[int]:
        observed_batch_sizes.append(len(rows))
        return [value * 10 for value in rows]

    batcher = DynamicBatcher(predict, max_batch_size=8, max_wait_ms=20)
    worker = asyncio.create_task(batcher.serve())
    try:
        outputs = await asyncio.gather(*(batcher.predict(i) for i in range(7)))
        await batcher.queue.join()
    finally:
        await stop_worker(worker)

    assert outputs == [i * 10 for i in range(7)]
    assert max(observed_batch_sizes) > 1


@pytest.mark.asyncio
async def test_bad_predictor_output_count_fails_request_without_killing_worker() -> None:
    calls = 0

    def predict(rows: list[int]) -> list[int]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return []
        return [value * 2 for value in rows]

    batcher = DynamicBatcher(predict, max_batch_size=4, max_wait_ms=1)
    worker = asyncio.create_task(batcher.serve())
    try:
        with pytest.raises(BatchContractError, match="one output per input"):
            await batcher.predict(3)
        assert await batcher.predict(4) == 8
        await batcher.queue.join()
    finally:
        await stop_worker(worker)


@pytest.mark.asyncio
async def test_predictor_exception_is_propagated_and_next_batch_recovers() -> None:
    def predict(rows: list[int]) -> list[int]:
        if any(value < 0 for value in rows):
            raise ValueError("synthetic predictor failure")
        return [value + 1 for value in rows]

    batcher = DynamicBatcher(predict, max_batch_size=4, max_wait_ms=1)
    worker = asyncio.create_task(batcher.serve())
    try:
        with pytest.raises(ValueError, match="synthetic predictor failure"):
            await batcher.predict(-1)
        assert await batcher.predict(10) == 11
        await batcher.queue.join()
    finally:
        await stop_worker(worker)


def test_batcher_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="max_batch_size"):
        DynamicBatcher(lambda rows: rows, max_batch_size=0)
    with pytest.raises(ValueError, match="max_wait_ms"):
        DynamicBatcher(lambda rows: rows, max_wait_ms=-1)


def test_model_rejects_wrong_feature_dimension() -> None:
    model = LinearRiskModel()
    with pytest.raises(ValueError, match="expected 4 features"):
        model.predict_batch([[1.0, 2.0]])


@pytest.mark.asyncio
async def test_prediction_endpoint_with_lifespan() -> None:
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            ready = await client.get("/ready")
            assert ready.status_code == 200

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
            payload = responses[0].json()
            assert 0 <= payload["score"] <= 1
            assert payload["label"] in {0, 1}
            assert payload["model"] == "synthetic-risk-model"

            metrics = await client.get("/metrics")
            assert metrics.status_code == 200
            assert "inference_requests_total" in metrics.text
