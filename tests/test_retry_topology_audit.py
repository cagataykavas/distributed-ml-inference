from __future__ import annotations

import json
from copy import deepcopy

import pytest

from retry_topology_audit import PolicyInputError, audit_policy, main, parse_policy


def safe_policy() -> dict[str, object]:
    return {
        "schema_version": 1,
        "generated_at": "2026-10-01T12:00:00Z",
        "config_digest": "a" * 64,
        "operation": {
            "id": "inference.predict",
            "method": "POST",
            "deadline_ms": 2_000,
            "idempotency_key_bound": True,
        },
        "limits": {
            "max_amplification": 2,
            "max_retry_layers": 1,
            "max_retry_budget_percent": 20,
        },
        "layers": [
            {
                "id": "service-client",
                "max_attempts": 2,
                "per_attempt_timeout_ms": 700,
                "retry_budget_percent": 10,
                "backoff_base_ms": 50,
                "backoff_cap_ms": 200,
                "jitter": "full",
                "retry_on": ["connect_error", "reset", "unavailable"],
                "honor_retry_after": True,
            }
        ],
    }


def parse(data: dict[str, object]):
    return parse_policy(json.dumps(data).encode())


def codes(report: dict[str, object]) -> set[str]:
    return {item["code"] for item in report["violations"]}  # type: ignore[index]


def test_safe_policy_reports_bounded_amplification_and_window() -> None:
    report = audit_policy(parse(safe_policy()))

    assert report["decision"] == "pass"
    assert report["metrics"] == {
        "effective_attempts": 2,
        "retry_layers": 1,
        "worst_case_window_ms": 1_450,
    }
    assert report["violations"] == []


def test_layered_retries_expose_multiplicative_amplification() -> None:
    data = safe_policy()
    outer = deepcopy(data["layers"][0])  # type: ignore[index]
    outer.update({"id": "edge", "per_attempt_timeout_ms": 1_500})
    data["layers"] = [outer, data["layers"][0]]  # type: ignore[index]

    report = audit_policy(parse(data))

    assert report["metrics"]["effective_attempts"] == 4  # type: ignore[index]
    assert {"retry_amplification_exceeded", "retry_layer_limit_exceeded"} <= codes(report)


def test_timeout_hierarchy_rejects_truncated_inner_retries() -> None:
    data = safe_policy()
    outer = {
        "id": "edge",
        "max_attempts": 1,
        "per_attempt_timeout_ms": 1_000,
        "retry_budget_percent": 0,
        "backoff_base_ms": 0,
        "backoff_cap_ms": 0,
        "jitter": "none",
        "retry_on": [],
        "honor_retry_after": False,
    }
    data["layers"] = [outer, data["layers"][0]]  # type: ignore[index]

    assert "timeout_hierarchy_invalid" in codes(audit_policy(parse(data)))


def test_retry_window_must_fit_end_to_end_deadline() -> None:
    data = safe_policy()
    data["operation"]["deadline_ms"] = 1_400  # type: ignore[index]

    assert "retry_window_exceeds_deadline" in codes(audit_policy(parse(data)))


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("retry_budget_percent", 0, "missing_retry_budget"),
        ("retry_budget_percent", 21, "retry_budget_exceeded"),
        ("jitter", "none", "synchronized_backoff"),
        ("backoff_base_ms", 0, "invalid_backoff"),
        ("retry_on", [], "missing_retry_conditions"),
    ],
)
def test_retry_controller_safety_requirements(field: str, value: object, expected: str) -> None:
    data = safe_policy()
    data["layers"][0][field] = value  # type: ignore[index]

    assert expected in codes(audit_policy(parse(data)))


@pytest.mark.parametrize("condition", ["rate_limited", "unavailable"])
def test_server_backpressure_retry_must_honor_retry_after(condition: str) -> None:
    data = safe_policy()
    data["layers"][0].update(  # type: ignore[index]
        {"retry_on": [condition], "honor_retry_after": False}
    )

    assert "retry_after_ignored" in codes(audit_policy(parse(data)))


def test_non_idempotent_retry_requires_bound_key() -> None:
    data = safe_policy()
    data["operation"]["idempotency_key_bound"] = False  # type: ignore[index]

    assert "unsafe_non_idempotent_retry" in codes(audit_policy(parse(data)))


def test_idempotent_method_does_not_require_bound_key() -> None:
    data = safe_policy()
    data["operation"].update({"method": "GET", "idempotency_key_bound": False})  # type: ignore[union-attr]

    assert "unsafe_non_idempotent_retry" not in codes(audit_policy(parse(data)))


def test_single_attempt_layer_rejects_dormant_retry_configuration() -> None:
    data = safe_policy()
    data["layers"][0]["max_attempts"] = 1  # type: ignore[index]

    assert "inactive_retry_configuration" in codes(audit_policy(parse(data)))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update({"unexpected": True}), "unknown keys"),
        (lambda data: data.update({"schema_version": True}), "schema_version"),
        (lambda data: data.update({"config_digest": "not-a-digest"}), "SHA-256"),
        (lambda data: data.update({"generated_at": "2026-10-01T12:00:00"}), "UTC offset"),
        (
            lambda data: data["layers"][0].update({"retry_on": ["application_error"]}),
            "unsafe conditions",
        ),
    ],
)
def test_parser_rejects_ambiguous_or_unsafe_input(mutation, message: str) -> None:
    data = safe_policy()
    mutation(data)

    with pytest.raises(PolicyInputError, match=message):
        parse(data)


def test_parser_rejects_duplicate_json_keys() -> None:
    payload = b'{"schema_version": 1, "schema_version": 1}'

    with pytest.raises(PolicyInputError, match="duplicate JSON key"):
        parse_policy(payload)


def test_parser_rejects_duplicate_layer_ids() -> None:
    data = safe_policy()
    data["layers"] = [data["layers"][0], deepcopy(data["layers"][0])]  # type: ignore[index]

    with pytest.raises(PolicyInputError, match="layer ids must be unique"):
        parse(data)


def test_parser_rejects_oversized_policy() -> None:
    with pytest.raises(PolicyInputError, match="exceeds"):
        parse_policy(b" " * (128 * 1024 + 1))


def test_cli_writes_failed_decision_and_returns_one(tmp_path) -> None:
    data = safe_policy()
    data["operation"]["idempotency_key_bound"] = False  # type: ignore[index]
    source = tmp_path / "policy.json"
    output = tmp_path / "decision.json"
    source.write_text(json.dumps(data), encoding="utf-8")

    assert main([str(source), "--output", str(output)]) == 1
    assert json.loads(output.read_text())["decision"] == "fail"


def test_cli_distinguishes_input_error_from_policy_failure(tmp_path, capsys) -> None:
    source = tmp_path / "policy.json"
    output = tmp_path / "decision.json"
    source.write_text("{}", encoding="utf-8")

    assert main([str(source), "--output", str(output)]) == 2
    assert "input error" in capsys.readouterr().err
    assert not output.exists()
