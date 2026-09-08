from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

from inference.model import ModelAdapter


@dataclass(frozen=True, slots=True)
class ModelSnapshot:
    generation: int
    adapter: ModelAdapter
    loaded_at: datetime

    def metadata(self) -> dict[str, int | str]:
        return {
            "generation": self.generation,
            "name": self.adapter.name,
            "version": self.adapter.version,
            "loaded_at": self.loaded_at.isoformat(),
        }


class ModelManager:
    """Atomically swaps validated models without mutating in-flight batches."""

    def __init__(self, adapter: ModelAdapter, *, feature_count: int = 4) -> None:
        self._feature_count = feature_count
        self._snapshot = ModelSnapshot(1, adapter, datetime.now(UTC))
        self._reload_lock = asyncio.Lock()

    @property
    def snapshot(self) -> ModelSnapshot:
        return self._snapshot

    def predict_batch(self, rows: list[list[float]]) -> list[dict[str, float | int | str]]:
        snapshot = self._snapshot
        return snapshot.adapter.predict_batch(rows)

    async def reload(self, candidate: ModelAdapter) -> ModelSnapshot:
        async with self._reload_lock:
            outputs = await asyncio.to_thread(
                candidate.predict_batch, [[0.0] * self._feature_count]
            )
            if len(outputs) != 1:
                raise ValueError("candidate model failed warm-up output contract")
            previous = self._snapshot
            self._snapshot = ModelSnapshot(
                generation=previous.generation + 1,
                adapter=candidate,
                loaded_at=datetime.now(UTC),
            )
            return self._snapshot
