from __future__ import annotations

import os
from dataclasses import dataclass


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    max_batch_size: int = 32
    max_wait_ms: int = 5
    max_queue_size: int = 256
    request_timeout_ms: int = 2_000
    shutdown_timeout_ms: int = 5_000
    circuit_failure_threshold: int = 5
    circuit_recovery_ms: int = 30_000
    admin_token: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            max_batch_size=_positive_int("INFERENCE_MAX_BATCH_SIZE", 32),
            max_wait_ms=_positive_int("INFERENCE_MAX_WAIT_MS", 5),
            max_queue_size=_positive_int("INFERENCE_MAX_QUEUE_SIZE", 256),
            request_timeout_ms=_positive_int("INFERENCE_REQUEST_TIMEOUT_MS", 2_000),
            shutdown_timeout_ms=_positive_int("INFERENCE_SHUTDOWN_TIMEOUT_MS", 5_000),
            circuit_failure_threshold=_positive_int("INFERENCE_CIRCUIT_FAILURE_THRESHOLD", 5),
            circuit_recovery_ms=_positive_int("INFERENCE_CIRCUIT_RECOVERY_MS", 30_000),
            admin_token=os.getenv("INFERENCE_ADMIN_TOKEN"),
        )
