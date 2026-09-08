from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Generic, TypeVar, cast

T = TypeVar("T")
R = TypeVar("R")
BatchPredictor = Callable[[list[T]], list[R] | Awaitable[list[R]]]


class BatchContractError(RuntimeError):
    """The predictor did not preserve the one-input/one-output contract."""


class BatcherClosed(RuntimeError):
    """The runtime cannot accept work in its current lifecycle state."""


class QueueOverloaded(RuntimeError):
    """The bounded admission queue has no capacity."""


class InferenceTimeout(TimeoutError):
    """A request exceeded its end-to-end deadline."""


class BatcherState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(slots=True)
class Request(Generic[T, R]):
    payload: T
    future: asyncio.Future[R]
    enqueued_at: float


@dataclass(slots=True)
class BatcherStats:
    accepted: int = 0
    completed: int = 0
    rejected: int = 0
    cancelled: int = 0
    timed_out: int = 0
    failed: int = 0
    batches: int = 0
    batch_items: int = 0


class DynamicBatcher(Generic[T, R]):
    """Lifecycle-safe, bounded dynamic batching runtime.

    Admission is deliberately non-blocking: callers receive ``QueueOverloaded``
    instead of waiting in an unbounded second queue. Synchronous predictors run
    in a worker thread so CPU/model latency never freezes the ASGI event loop.
    """

    def __init__(
        self,
        predict_batch: BatchPredictor[T, R],
        max_batch_size: int = 32,
        max_wait_ms: int = 5,
        max_queue_size: int = 256,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self.predict_batch = predict_batch
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000
        self.queue: asyncio.Queue[Request[T, R]] = asyncio.Queue(maxsize=max_queue_size)
        self.state = BatcherState.CREATED
        self.stats = BatcherStats()
        self._worker: asyncio.Task[None] | None = None
        self._in_flight = 0

    @property
    def is_ready(self) -> bool:
        return self.state is BatcherState.RUNNING and (
            self._worker is None or not self._worker.done()
        )

    def snapshot(self) -> dict[str, int | str]:
        return {
            "state": self.state.value,
            "queue_depth": self.queue.qsize(),
            "queue_capacity": self.queue.maxsize,
            "in_flight": self._in_flight,
            **asdict(self.stats),
        }

    async def start(self) -> None:
        if self.state is not BatcherState.CREATED:
            raise BatcherClosed(f"cannot start batcher in state {self.state.value}")
        self.state = BatcherState.RUNNING
        self._worker = asyncio.create_task(self.serve(), name="dynamic-inference-batcher")
        await asyncio.sleep(0)

    async def predict(self, payload: T, *, timeout_seconds: float | None = None) -> R:
        if not self.is_ready:
            raise BatcherClosed(f"batcher is {self.state.value}")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[R] = loop.create_future()
        request = Request(payload=payload, future=future, enqueued_at=loop.time())
        try:
            self.queue.put_nowait(request)
        except asyncio.QueueFull as exc:
            self.stats.rejected += 1
            raise QueueOverloaded("inference admission queue is full") from exc
        self.stats.accepted += 1
        try:
            if timeout_seconds is None:
                return await future
            async with asyncio.timeout(timeout_seconds):
                return await asyncio.shield(future)
        except TimeoutError as exc:
            self.stats.timed_out += 1
            future.cancel()
            raise InferenceTimeout("inference request deadline exceeded") from exc
        except asyncio.CancelledError:
            self.stats.cancelled += 1
            future.cancel()
            raise

    async def close(self, *, drain: bool = True, timeout_seconds: float = 5.0) -> None:
        if self.state is BatcherState.CLOSED:
            return
        self.state = BatcherState.CLOSING
        if drain and self._worker is not None and not self._worker.done():
            try:
                await asyncio.wait_for(self.queue.join(), timeout=timeout_seconds)
            except TimeoutError:
                pass
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        self._reject_queued(BatcherClosed("batcher closed before request execution"))
        self.state = BatcherState.CLOSED

    async def _execute(self, payloads: list[T]) -> list[R]:
        if inspect.iscoroutinefunction(self.predict_batch):
            return await cast(Awaitable[list[R]], self.predict_batch(payloads))
        predictor = cast(Callable[[list[T]], list[R]], self.predict_batch)
        return await asyncio.to_thread(predictor, payloads)

    async def _resolve_batch(self, batch: list[Request[T, R]]) -> None:
        active = [request for request in batch if not request.future.done()]
        if not active:
            return
        self._in_flight += len(active)
        self.stats.batches += 1
        self.stats.batch_items += len(active)
        try:
            outputs = await self._execute([request.payload for request in active])
            if len(outputs) != len(active):
                raise BatchContractError(
                    "predict_batch must return exactly one output per input: "
                    f"received {len(outputs)} outputs for {len(active)} inputs"
                )
            for request, output in zip(active, outputs, strict=True):
                if not request.future.done():
                    request.future.set_result(output)
                    self.stats.completed += 1
        except Exception as exc:  # noqa: BLE001 - model boundary fails the complete batch.
            for request in active:
                if not request.future.done():
                    request.future.set_exception(exc)
                    self.stats.failed += 1
        finally:
            self._in_flight -= len(active)

    def _reject_queued(self, error: Exception) -> None:
        while True:
            try:
                request = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if not request.future.done():
                request.future.set_exception(error)
            self.queue.task_done()

    async def serve(self) -> None:
        if self.state is BatcherState.CREATED:
            self.state = BatcherState.RUNNING
        try:
            while self.state is BatcherState.RUNNING or (
                self.state is BatcherState.CLOSING and not self.queue.empty()
            ):
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
                    await self._resolve_batch(batch)
                finally:
                    for _ in batch:
                        self.queue.task_done()
        except asyncio.CancelledError:
            self._reject_queued(BatcherClosed("batch worker stopped"))
            raise
        finally:
            if self.state is BatcherState.RUNNING:
                self.state = BatcherState.CLOSED


async def demo() -> None:
    batcher = DynamicBatcher(lambda xs: [x * x for x in xs], max_batch_size=8)
    await batcher.start()
    try:
        print(await asyncio.gather(*(batcher.predict(i) for i in range(20))))
    finally:
        await batcher.close()


if __name__ == "__main__":
    asyncio.run(demo())
