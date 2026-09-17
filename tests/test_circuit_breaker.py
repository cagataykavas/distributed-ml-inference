from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from inference.circuit_breaker import CircuitBreaker, CircuitOpen


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_consecutive_failures_open_and_success_resets() -> None:
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=10, clock=clock)

    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.before_call()
    breaker.record_failure()

    with pytest.raises(CircuitOpen) as error:
        breaker.before_call()
    assert error.value.retry_after_seconds == 10
    assert breaker.snapshot() == {
        "state": "open",
        "consecutive_failures": 2,
        "rejected_calls": 1,
        "opened_total": 1,
        "probe_in_flight": False,
    }


def test_half_open_probe_closes_breaker_after_success() -> None:
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=5, clock=clock)
    breaker.record_failure()
    clock.now = 5

    breaker.before_call()
    assert breaker.snapshot()["state"] == "half_open"
    with pytest.raises(CircuitOpen):
        breaker.before_call()

    breaker.record_success()
    breaker.before_call()
    assert breaker.snapshot()["state"] == "closed"


def test_failed_probe_reopens_for_a_full_recovery_window() -> None:
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=5, clock=clock)
    breaker.record_failure()
    clock.now = 5
    breaker.before_call()
    breaker.record_failure()

    clock.now = 9
    with pytest.raises(CircuitOpen) as error:
        breaker.before_call()
    assert error.value.retry_after_seconds == 1
    assert breaker.snapshot()["opened_total"] == 2


def test_only_one_concurrent_half_open_probe_is_admitted() -> None:
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=1, clock=clock)
    breaker.record_failure()
    clock.now = 1

    def attempt() -> bool:
        try:
            breaker.before_call()
        except CircuitOpen:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        admitted = list(pool.map(lambda _: attempt(), range(32)))

    assert sum(admitted) == 1
    assert breaker.snapshot()["rejected_calls"] == 31


@pytest.mark.parametrize(
    ("threshold", "timeout"),
    [(0, 1), (1, 0), (-1, 1), (1, -1)],
)
def test_invalid_configuration_is_rejected(threshold: int, timeout: float) -> None:
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=threshold, recovery_timeout_seconds=timeout)
