from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

import pytest

from batcher import DynamicBatcher
from inference.model import LinearRiskModel
from inference.runtime import ModelManager


@dataclass
class BlockingModel:
    entered: threading.Event
    release: threading.Event
    name: str = "blocking"
    version: str = "old"

    def predict_batch(self, rows: list[list[float]]) -> list[dict[str, float | int | str]]:
        self.entered.set()
        self.release.wait(timeout=1)
        return [
            {"score": 0.1, "label": 0, "model": self.name, "version": self.version} for _ in rows
        ]


@pytest.mark.asyncio
async def test_atomic_reload_preserves_in_flight_model_generation() -> None:
    entered = threading.Event()
    release = threading.Event()
    manager = ModelManager(BlockingModel(entered, release))
    batcher = DynamicBatcher(manager.predict_batch, max_batch_size=1, max_wait_ms=0)
    await batcher.start()
    old_request = asyncio.create_task(batcher.predict([[0.0, 0.0, 0.0, 0.0]][0]))
    assert await asyncio.to_thread(entered.wait, 0.5)

    snapshot = await manager.reload(LinearRiskModel(version="new"))
    release.set()

    old_result = await old_request
    new_result = await batcher.predict([0.0, 0.0, 0.0, 0.0])
    await batcher.close()
    assert old_result["version"] == "old"
    assert new_result["version"] == "new"
    assert snapshot.generation == 2


@pytest.mark.asyncio
async def test_invalid_candidate_does_not_replace_active_model() -> None:
    manager = ModelManager(LinearRiskModel(version="stable"))
    invalid = LinearRiskModel(weights=(float("nan"), 0.0, 0.0, 0.0), version="bad")
    with pytest.raises(ValueError, match="finite"):
        await manager.reload(invalid)
    assert manager.snapshot.adapter.version == "stable"
    assert manager.snapshot.generation == 1


def test_model_rejects_invalid_features() -> None:
    model = LinearRiskModel()
    with pytest.raises(ValueError, match="expected 4 features"):
        model.predict_batch([[1.0, 2.0]])
    with pytest.raises(ValueError, match="finite"):
        model.predict_batch([[float("inf"), 0.0, 0.0, 0.0]])
