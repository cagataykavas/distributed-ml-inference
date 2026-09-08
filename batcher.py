from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class BatchContractError(RuntimeError):
    """Raised when the configured batch predictor violates the one-in/one-out contract."""


@dataclass
class Request(Generic[T, R]):
    payload: T
    future: asyncio.Future[R]


class DynamicBatcher(Generic[T, R]):
    """Collect concurrent inference requests into bounded micro-batches."""

    def __init__(
        self,
        predict_batch: Callable[[list[T]], list[R]],
        max_batch_size: int = 32,
        max_wait_ms: int = 5,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")

        self.predict_batch = predict_batch
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000
        self.queue: asyncio.Queue[Request[T, R]] = asyncio.Queue()

    async def predict(self, payload: T) -> R:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[R] = loop.create_future()
        await self.queue.put(Request(payload, future))
        return await future

    def _resolve_batch(self, batch: list[Request[T, R]]) -> None:
        try:
            outputs = self.predict_batch([request.payload for request in batch])
            if len(outputs) != len(batch):
                raise BatchContractError(
                    "predict_batch must return exactly one output per input: "
                    f"received {len(outputs)} outputs for {len(batch)} inputs"
                )
            for request, output in zip(batch, outputs):
                if not request.future.done():
                    request.future.set_result(output)
        except Exception as exc:  # noqa: BLE001 - predictor boundary must fail every request in a batch.
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(exc)

    async def serve(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            try:
                deadline = asyncio.get_running_loop().time() + self.max_wait
                while len(batch) < self.max_batch_size:
                    timeout = deadline - asyncio.get_running_loop().time()
                    if timeout <= 0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self.queue.get(), timeout))
                    except TimeoutError:
                        break

                self._resolve_batch(batch)
            except asyncio.CancelledError:
                for request in batch:
                    if not request.future.done():
                        request.future.cancel()
                raise
            finally:
                for _ in batch:
                    self.queue.task_done()


async def demo() -> None:
    batcher = DynamicBatcher(lambda xs: [x * x for x in xs], max_batch_size=8, max_wait_ms=10)
    worker = asyncio.create_task(batcher.serve())
    try:
        print(await asyncio.gather(*(batcher.predict(i) for i in range(20))))
        await batcher.queue.join()
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(demo())
