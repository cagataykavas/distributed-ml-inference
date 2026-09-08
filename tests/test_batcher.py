from __future__ import annotations

import asyncio
import threading

import pytest

from batcher import (
    BatchContractError,
    BatcherClosed,
    BatcherState,
    DynamicBatcher,
    InferenceTimeout,
    QueueOverloaded,
)


@pytest.mark.asyncio
async def test_dynamic_batcher_preserves_order_and_batches_requests() -> None:
    sizes: list[int] = []

    def predict(rows: list[int]) -> list[int]:
        sizes.append(len(rows))
        return [value * 10 for value in rows]

    batcher = DynamicBatcher(predict, max_batch_size=8, max_wait_ms=20)
    await batcher.start()
    try:
        outputs = await asyncio.gather(*(batcher.predict(i) for i in range(7)))
    finally:
        await batcher.close()

    assert outputs == [i * 10 for i in range(7)]
    assert max(sizes) > 1
    assert batcher.snapshot()["completed"] == 7


@pytest.mark.asyncio
async def test_bounded_queue_rejects_overload_without_hidden_waiters() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow(rows: list[int]) -> list[int]:
        entered.set()
        await release.wait()
        return rows

    batcher = DynamicBatcher(slow, max_batch_size=1, max_wait_ms=0, max_queue_size=1)
    await batcher.start()
    first = asyncio.create_task(batcher.predict(1))
    await entered.wait()
    second = asyncio.create_task(batcher.predict(2))
    await asyncio.sleep(0)

    with pytest.raises(QueueOverloaded):
        await batcher.predict(3)

    release.set()
    assert await asyncio.gather(first, second) == [1, 2]
    assert batcher.snapshot()["rejected"] == 1
    await batcher.close()


@pytest.mark.asyncio
async def test_cancelled_queued_request_is_not_sent_to_predictor() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    observed: list[int] = []

    async def slow(rows: list[int]) -> list[int]:
        observed.extend(rows)
        entered.set()
        await release.wait()
        return rows

    batcher = DynamicBatcher(slow, max_batch_size=1, max_wait_ms=0, max_queue_size=2)
    await batcher.start()
    first = asyncio.create_task(batcher.predict(1))
    await entered.wait()
    cancelled = asyncio.create_task(batcher.predict(2))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    release.set()
    assert await first == 1
    await batcher.queue.join()
    await batcher.close()
    assert observed == [1]
    assert batcher.snapshot()["cancelled"] == 1


@pytest.mark.asyncio
async def test_deadline_expires_and_worker_recovers() -> None:
    async def slow(rows: list[int]) -> list[int]:
        await asyncio.sleep(0.03)
        return rows

    batcher = DynamicBatcher(slow, max_batch_size=1, max_wait_ms=0)
    await batcher.start()
    with pytest.raises(InferenceTimeout):
        await batcher.predict(1, timeout_seconds=0.001)
    assert await batcher.predict(2, timeout_seconds=0.2) == 2
    await batcher.close()
    assert batcher.snapshot()["timed_out"] == 1


@pytest.mark.asyncio
async def test_sync_predictor_does_not_block_event_loop() -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking(rows: list[int]) -> list[int]:
        entered.set()
        release.wait(timeout=1)
        return rows

    batcher = DynamicBatcher(blocking, max_batch_size=1, max_wait_ms=0)
    await batcher.start()
    prediction = asyncio.create_task(batcher.predict(7))
    assert await asyncio.to_thread(entered.wait, 0.5)
    ticked = False

    async def ticker() -> None:
        nonlocal ticked
        await asyncio.sleep(0.005)
        ticked = True

    await ticker()
    assert ticked
    release.set()
    assert await prediction == 7
    await batcher.close()


@pytest.mark.asyncio
async def test_contract_failure_does_not_kill_worker() -> None:
    calls = 0

    def predict(rows: list[int]) -> list[int]:
        nonlocal calls
        calls += 1
        return [] if calls == 1 else [value * 2 for value in rows]

    batcher = DynamicBatcher(predict, max_batch_size=1, max_wait_ms=0)
    await batcher.start()
    with pytest.raises(BatchContractError, match="one output per input"):
        await batcher.predict(3)
    assert await batcher.predict(4) == 8
    await batcher.close()
    assert batcher.snapshot()["failed"] == 1


@pytest.mark.asyncio
async def test_closed_runtime_rejects_new_work() -> None:
    batcher = DynamicBatcher(lambda rows: rows)
    await batcher.start()
    await batcher.close()
    assert batcher.state is BatcherState.CLOSED
    with pytest.raises(BatcherClosed):
        await batcher.predict(1)


def test_batcher_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="max_batch_size"):
        DynamicBatcher(lambda rows: rows, max_batch_size=0)
    with pytest.raises(ValueError, match="max_wait_ms"):
        DynamicBatcher(lambda rows: rows, max_wait_ms=-1)
    with pytest.raises(ValueError, match="max_queue_size"):
        DynamicBatcher(lambda rows: rows, max_queue_size=0)
