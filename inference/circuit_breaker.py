from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from threading import Lock


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpen(RuntimeError):
    """Inference is temporarily blocked while the model dependency recovers."""

    def __init__(self, retry_after_seconds: float) -> None:
        super().__init__("inference circuit is open")
        self.retry_after_seconds = max(0.0, retry_after_seconds)


@dataclass(frozen=True, slots=True)
class CircuitSnapshot:
    state: str
    consecutive_failures: int
    rejected_calls: int
    opened_total: int
    probe_in_flight: bool


class CircuitBreaker:
    """Thread-safe consecutive-failure breaker with a single half-open probe."""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if recovery_timeout_seconds <= 0:
            raise ValueError("recovery_timeout_seconds must be positive")
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self._clock = clock
        self._lock = Lock()
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False
        self._rejected_calls = 0
        self._opened_total = 0

    def before_call(self) -> None:
        """Acquire permission for a model call or fail fast.

        Once the recovery window elapses, exactly one caller becomes the
        half-open probe. Concurrent callers continue to fail fast.
        """
        with self._lock:
            if self._state is CircuitState.CLOSED:
                return
            now = self._clock()
            assert self._opened_at is not None
            remaining = self.recovery_timeout_seconds - (now - self._opened_at)
            if self._state is CircuitState.OPEN and remaining <= 0:
                self._state = CircuitState.HALF_OPEN
                self._probe_in_flight = True
                return
            self._rejected_calls += 1
            raise CircuitOpen(max(0.0, remaining))

    def record_success(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if (
                self._state is CircuitState.HALF_OPEN
                or self._consecutive_failures >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                self._probe_in_flight = False
                self._opened_total += 1

    @property
    def is_available(self) -> bool:
        with self._lock:
            if self._state is CircuitState.CLOSED:
                return True
            if self._state is CircuitState.OPEN and self._opened_at is not None:
                return self._clock() - self._opened_at >= self.recovery_timeout_seconds
            return False

    def snapshot(self) -> dict[str, int | str | bool]:
        with self._lock:
            return asdict(
                CircuitSnapshot(
                    state=self._state.value,
                    consecutive_failures=self._consecutive_failures,
                    rejected_calls=self._rejected_calls,
                    opened_total=self._opened_total,
                    probe_in_flight=self._probe_in_flight,
                )
            )
