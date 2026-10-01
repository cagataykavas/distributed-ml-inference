"""Fail-closed audit for layered retry policies.

The input describes retry controllers from the outermost caller to the
innermost transport. The audit makes multiplicative retry amplification and
deadline mismatches visible before a policy reaches production.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

MAX_POLICY_BYTES = 128 * 1024
MAX_LAYERS = 16
MAX_ATTEMPTS_PER_LAYER = 10
MAX_DEADLINE_MS = 300_000
SAFE_RETRY_CONDITIONS = frozenset(
    {
        "connect_error",
        "connect_timeout",
        "rate_limited",
        "reset",
        "unavailable",
    }
)
IDEMPOTENT_METHODS = frozenset({"DELETE", "GET", "HEAD", "OPTIONS", "PUT"})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PolicyInputError(ValueError):
    """Raised when an artifact is malformed or outside parser bounds."""


@dataclass(frozen=True, slots=True)
class Operation:
    operation_id: str
    method: str
    deadline_ms: int
    idempotency_key_bound: bool


@dataclass(frozen=True, slots=True)
class Limits:
    max_amplification: int
    max_retry_layers: int
    max_retry_budget_percent: int


@dataclass(frozen=True, slots=True)
class RetryLayer:
    layer_id: str
    max_attempts: int
    per_attempt_timeout_ms: int
    retry_budget_percent: int
    backoff_base_ms: int
    backoff_cap_ms: int
    jitter: str
    retry_on: tuple[str, ...]
    honor_retry_after: bool

    @property
    def retries(self) -> bool:
        return self.max_attempts > 1

    def maximum_backoff_ms(self) -> int:
        """Return the bounded maximum delay across retry gaps."""
        if not self.retries or self.backoff_base_ms <= 0:
            return 0
        return sum(
            min(self.backoff_cap_ms, self.backoff_base_ms * (2**retry_index))
            for retry_index in range(self.max_attempts - 1)
        )


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    generated_at: str
    config_digest: str
    operation: Operation
    limits: Limits
    layers: tuple[RetryLayer, ...]


@dataclass(frozen=True, slots=True)
class Violation:
    code: str
    message: str
    layer_id: str | None = None

    def as_dict(self) -> dict[str, str]:
        payload = {"code": self.code, "message": self.message}
        if self.layer_id is not None:
            payload["layer_id"] = self.layer_id
        return payload


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyInputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PolicyInputError(f"{context} must be an object")
    return value


def _expect_keys(
    value: dict[str, Any],
    *,
    context: str,
    required: set[str],
) -> None:
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required)
    if missing:
        raise PolicyInputError(f"{context} missing keys: {', '.join(missing)}")
    if unknown:
        raise PolicyInputError(f"{context} has unknown keys: {', '.join(unknown)}")


def _integer(value: Any, context: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise PolicyInputError(f"{context} must be an integer in [{minimum}, {maximum}]")
    return value


def _boolean(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise PolicyInputError(f"{context} must be a boolean")
    return value


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise PolicyInputError(f"{context} must be a bounded identifier")
    return value


def _parse_timestamp(value: Any) -> str:
    if not isinstance(value, str):
        raise PolicyInputError("generated_at must be an RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PolicyInputError("generated_at must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PolicyInputError("generated_at must include an explicit UTC offset")
    return value


def _parse_operation(value: Any) -> Operation:
    data = _mapping(value, "operation")
    _expect_keys(
        data,
        context="operation",
        required={"id", "method", "deadline_ms", "idempotency_key_bound"},
    )
    method = data["method"]
    if not isinstance(method, str) or method.upper() not in {
        "DELETE",
        "GET",
        "HEAD",
        "OPTIONS",
        "PATCH",
        "POST",
        "PUT",
    }:
        raise PolicyInputError("operation.method is not a supported HTTP method")
    return Operation(
        operation_id=_identifier(data["id"], "operation.id"),
        method=method.upper(),
        deadline_ms=_integer(
            data["deadline_ms"], "operation.deadline_ms", minimum=1, maximum=MAX_DEADLINE_MS
        ),
        idempotency_key_bound=_boolean(
            data["idempotency_key_bound"], "operation.idempotency_key_bound"
        ),
    )


def _parse_limits(value: Any) -> Limits:
    data = _mapping(value, "limits")
    _expect_keys(
        data,
        context="limits",
        required={"max_amplification", "max_retry_layers", "max_retry_budget_percent"},
    )
    return Limits(
        max_amplification=_integer(
            data["max_amplification"], "limits.max_amplification", minimum=1, maximum=1_000
        ),
        max_retry_layers=_integer(
            data["max_retry_layers"], "limits.max_retry_layers", minimum=0, maximum=MAX_LAYERS
        ),
        max_retry_budget_percent=_integer(
            data["max_retry_budget_percent"],
            "limits.max_retry_budget_percent",
            minimum=0,
            maximum=100,
        ),
    )


def _parse_layer(value: Any, index: int) -> RetryLayer:
    context = f"layers[{index}]"
    data = _mapping(value, context)
    _expect_keys(
        data,
        context=context,
        required={
            "id",
            "max_attempts",
            "per_attempt_timeout_ms",
            "retry_budget_percent",
            "backoff_base_ms",
            "backoff_cap_ms",
            "jitter",
            "retry_on",
            "honor_retry_after",
        },
    )
    jitter = data["jitter"]
    if jitter not in {"equal", "full", "none"}:
        raise PolicyInputError(f"{context}.jitter must be equal, full, or none")
    retry_on = data["retry_on"]
    if not isinstance(retry_on, list) or any(not isinstance(item, str) for item in retry_on):
        raise PolicyInputError(f"{context}.retry_on must be a string list")
    if len(retry_on) != len(set(retry_on)):
        raise PolicyInputError(f"{context}.retry_on contains duplicates")
    unsupported = sorted(set(retry_on) - SAFE_RETRY_CONDITIONS)
    if unsupported:
        raise PolicyInputError(
            f"{context}.retry_on has unsafe conditions: {', '.join(unsupported)}"
        )
    return RetryLayer(
        layer_id=_identifier(data["id"], f"{context}.id"),
        max_attempts=_integer(
            data["max_attempts"],
            f"{context}.max_attempts",
            minimum=1,
            maximum=MAX_ATTEMPTS_PER_LAYER,
        ),
        per_attempt_timeout_ms=_integer(
            data["per_attempt_timeout_ms"],
            f"{context}.per_attempt_timeout_ms",
            minimum=1,
            maximum=MAX_DEADLINE_MS,
        ),
        retry_budget_percent=_integer(
            data["retry_budget_percent"],
            f"{context}.retry_budget_percent",
            minimum=0,
            maximum=100,
        ),
        backoff_base_ms=_integer(
            data["backoff_base_ms"],
            f"{context}.backoff_base_ms",
            minimum=0,
            maximum=MAX_DEADLINE_MS,
        ),
        backoff_cap_ms=_integer(
            data["backoff_cap_ms"],
            f"{context}.backoff_cap_ms",
            minimum=0,
            maximum=MAX_DEADLINE_MS,
        ),
        jitter=jitter,
        retry_on=tuple(retry_on),
        honor_retry_after=_boolean(data["honor_retry_after"], f"{context}.honor_retry_after"),
    )


def parse_policy(payload: bytes) -> RetryPolicy:
    """Parse a bounded JSON policy and reject ambiguity."""
    if len(payload) > MAX_POLICY_BYTES:
        raise PolicyInputError(f"policy exceeds {MAX_POLICY_BYTES} bytes")
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PolicyInputError("policy must be UTF-8 JSON") from exc
    try:
        value = json.loads(decoded, object_pairs_hook=_object_without_duplicate_keys)
    except PolicyInputError:
        raise
    except json.JSONDecodeError as exc:
        raise PolicyInputError(f"invalid JSON at line {exc.lineno}, column {exc.colno}") from exc

    data = _mapping(value, "policy")
    _expect_keys(
        data,
        context="policy",
        required={
            "schema_version",
            "generated_at",
            "config_digest",
            "operation",
            "limits",
            "layers",
        },
    )
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise PolicyInputError("schema_version must be 1")
    digest = data["config_digest"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise PolicyInputError("config_digest must be a lowercase SHA-256 hex digest")
    layers_value = data["layers"]
    if not isinstance(layers_value, list) or not 1 <= len(layers_value) <= MAX_LAYERS:
        raise PolicyInputError(f"layers must contain between 1 and {MAX_LAYERS} entries")
    layers = tuple(_parse_layer(item, index) for index, item in enumerate(layers_value))
    layer_ids = [layer.layer_id for layer in layers]
    if len(layer_ids) != len(set(layer_ids)):
        raise PolicyInputError("layer ids must be unique")
    return RetryPolicy(
        generated_at=_parse_timestamp(data["generated_at"]),
        config_digest=digest,
        operation=_parse_operation(data["operation"]),
        limits=_parse_limits(data["limits"]),
        layers=layers,
    )


def _retry_window(policy: RetryPolicy, violations: list[Violation]) -> int:
    inner_window_ms: int | None = None
    for layer in reversed(policy.layers):
        if inner_window_ms is None:
            invocation_ms = layer.per_attempt_timeout_ms
        else:
            if inner_window_ms > layer.per_attempt_timeout_ms:
                violations.append(
                    Violation(
                        code="timeout_hierarchy_invalid",
                        layer_id=layer.layer_id,
                        message=(
                            f"inner retry window {inner_window_ms}ms exceeds this layer's "
                            f"{layer.per_attempt_timeout_ms}ms per-attempt timeout"
                        ),
                    )
                )
            invocation_ms = min(inner_window_ms, layer.per_attempt_timeout_ms)
        inner_window_ms = layer.max_attempts * invocation_ms + layer.maximum_backoff_ms()
    assert inner_window_ms is not None
    return inner_window_ms


def audit_policy(policy: RetryPolicy) -> dict[str, Any]:
    """Evaluate retry amplification, timing, and safety invariants."""
    violations: list[Violation] = []
    retry_layers = [layer for layer in policy.layers if layer.retries]
    amplification = math.prod(layer.max_attempts for layer in policy.layers)

    if len(retry_layers) > policy.limits.max_retry_layers:
        violations.append(
            Violation(
                code="retry_layer_limit_exceeded",
                message=(
                    f"{len(retry_layers)} retrying layers exceed the configured limit "
                    f"of {policy.limits.max_retry_layers}"
                ),
            )
        )
    if amplification > policy.limits.max_amplification:
        violations.append(
            Violation(
                code="retry_amplification_exceeded",
                message=(
                    f"worst-case amplification {amplification} exceeds the configured limit "
                    f"of {policy.limits.max_amplification}"
                ),
            )
        )

    for layer in policy.layers:
        if layer.retries:
            if layer.retry_budget_percent == 0:
                violations.append(
                    Violation(
                        "missing_retry_budget",
                        "a retrying layer must reserve a non-zero retry budget",
                        layer.layer_id,
                    )
                )
            elif layer.retry_budget_percent > policy.limits.max_retry_budget_percent:
                violations.append(
                    Violation(
                        "retry_budget_exceeded",
                        (
                            f"retry budget {layer.retry_budget_percent}% exceeds the configured "
                            f"limit of {policy.limits.max_retry_budget_percent}%"
                        ),
                        layer.layer_id,
                    )
                )
            if layer.backoff_base_ms == 0 or layer.backoff_cap_ms < layer.backoff_base_ms:
                violations.append(
                    Violation(
                        "invalid_backoff",
                        "retries require a positive base and a cap at least as large as the base",
                        layer.layer_id,
                    )
                )
            if layer.jitter == "none":
                violations.append(
                    Violation(
                        "synchronized_backoff",
                        "retries require equal or full jitter to avoid synchronized retry waves",
                        layer.layer_id,
                    )
                )
            if not layer.retry_on:
                violations.append(
                    Violation(
                        "missing_retry_conditions",
                        "a retrying layer must declare bounded retry conditions",
                        layer.layer_id,
                    )
                )
            if {"rate_limited", "unavailable"}.intersection(
                layer.retry_on
            ) and not layer.honor_retry_after:
                violations.append(
                    Violation(
                        "retry_after_ignored",
                        "rate-limited or unavailable retries must honor Retry-After",
                        layer.layer_id,
                    )
                )
        elif (
            any(
                (
                    layer.retry_budget_percent,
                    layer.backoff_base_ms,
                    layer.backoff_cap_ms,
                    len(layer.retry_on),
                    layer.honor_retry_after,
                )
            )
            or layer.jitter != "none"
        ):
            violations.append(
                Violation(
                    "inactive_retry_configuration",
                    "a single-attempt layer must not carry dormant retry settings",
                    layer.layer_id,
                )
            )

    if (
        retry_layers
        and policy.operation.method not in IDEMPOTENT_METHODS
        and not policy.operation.idempotency_key_bound
    ):
        violations.append(
            Violation(
                code="unsafe_non_idempotent_retry",
                message=(
                    f"{policy.operation.method} retries require an idempotency key bound to the "
                    "operation result"
                ),
            )
        )

    retry_window_ms = _retry_window(policy, violations)
    if retry_window_ms > policy.operation.deadline_ms:
        violations.append(
            Violation(
                code="retry_window_exceeds_deadline",
                message=(
                    f"worst-case retry window {retry_window_ms}ms exceeds the operation deadline "
                    f"of {policy.operation.deadline_ms}ms"
                ),
            )
        )

    ordered = sorted(
        violations,
        key=lambda violation: (violation.code, violation.layer_id or "", violation.message),
    )
    return {
        "schema_version": 1,
        "operation_id": policy.operation.operation_id,
        "generated_at": policy.generated_at,
        "config_digest": policy.config_digest,
        "decision": "pass" if not ordered else "fail",
        "metrics": {
            "effective_attempts": amplification,
            "retry_layers": len(retry_layers),
            "worst_case_window_ms": retry_window_ms,
        },
        "violations": [violation.as_dict() for violation in ordered],
    }


def _write_report(report: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit layered retries for amplification and deadline safety."
    )
    parser.add_argument("policy", type=Path, help="retry policy JSON")
    parser.add_argument("--output", type=Path, help="write the decision artifact to this path")
    args = parser.parse_args(argv)
    try:
        policy = parse_policy(args.policy.read_bytes())
        report = audit_policy(policy)
        _write_report(report, args.output)
    except (OSError, PolicyInputError) as exc:
        sys.stderr.write(f"retry policy input error: {exc}\n")
        return 2
    return 0 if report["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
