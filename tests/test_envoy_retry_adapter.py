from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from envoy_retry_adapter import (
    MAX_CONFIG_BYTES,
    EnvoyRetryAdapterError,
    adapt_envoy_retry_config,
    main,
)
from retry_topology_audit import audit_policy, parse_policy

GENERATED_AT = "2026-11-09T09:30:00Z"
OVERRIDE_HEADERS = [
    "x-envoy-hedge-on-per-try-timeout",
    "x-envoy-max-retries",
    "x-envoy-retriable-header-names",
    "x-envoy-retriable-status-codes",
    "x-envoy-retry-grpc-on",
    "x-envoy-retry-on",
    "x-envoy-upstream-rq-per-try-idle-timeout-ms",
    "x-envoy-upstream-rq-per-try-timeout-ms",
    "x-envoy-upstream-rq-timeout-ms",
]


def envoy_config() -> dict[str, object]:
    route = {
        "name": "inference.predict",
        "match": {
            "prefix": "/predict",
            "headers": [
                {"name": ":method", "exact_match": "POST"},
                {"name": "x-idempotency-key", "present_match": True},
            ],
        },
        "metadata": {"filter_metadata": {"retry_topology": {"idempotency_key_bound": True}}},
        "route": {
            "cluster": "inference_backend",
            "timeout": "2s",
            "retry_policy": {
                "retry_on": "connect-failure,reset,retriable-status-codes",
                "num_retries": {"value": 1},
                "per_try_timeout": "700ms",
                "retriable_status_codes": [503],
                "retry_back_off": {
                    "base_interval": "50ms",
                    "max_interval": "200ms",
                },
                "rate_limited_retry_back_off": {
                    "reset_headers": [{"name": "Retry-After", "format": "SECONDS"}],
                    "max_interval": "2s",
                },
            },
        },
    }
    return {
        "static_resources": {
            "listeners": [
                {
                    "name": "inference_listener",
                    "filter_chains": [
                        {
                            "filters": [
                                {
                                    "name": "envoy.filters.network.http_connection_manager",
                                    "typed_config": {
                                        "@type": (
                                            "type.googleapis.com/envoy.extensions.filters.network."
                                            "http_connection_manager.v3.HttpConnectionManager"
                                        ),
                                        "route_config": {
                                            "name": "inference_routes",
                                            "request_headers_to_remove": list(OVERRIDE_HEADERS),
                                            "virtual_hosts": [
                                                {
                                                    "name": "inference_api",
                                                    "domains": ["*"],
                                                    "routes": [route],
                                                }
                                            ],
                                        },
                                    },
                                }
                            ]
                        }
                    ],
                }
            ],
            "clusters": [
                {
                    "name": "inference_backend",
                    "type": "STRICT_DNS",
                    "circuit_breakers": {
                        "thresholds": [
                            {
                                "priority": "DEFAULT",
                                "retry_budget": {
                                    "budget_percent": {"value": 10},
                                    "budget_interval": "0s",
                                    "min_retry_concurrency": {"value": 0},
                                },
                            }
                        ]
                    },
                }
            ],
        }
    }


def adapt(config=None, **overrides):
    arguments = {
        "listener_name": "inference_listener",
        "virtual_host_name": "inference_api",
        "route_name": "inference.predict",
        "generated_at": GENERATED_AT,
        "max_amplification": 2,
        "max_retry_layers": 1,
        "max_retry_budget_percent": 20,
    }
    arguments.update(overrides)
    return adapt_envoy_retry_config(envoy_config() if config is None else config, **arguments)


def selected_route(config: dict[str, object]) -> dict[str, object]:
    return config["static_resources"]["listeners"][0]["filter_chains"][0]["filters"][0][
        "typed_config"
    ]["route_config"]["virtual_hosts"][0]["routes"][0]


def retry_policy(config: dict[str, object]) -> dict[str, object]:
    return selected_route(config)["route"]["retry_policy"]


def default_threshold(config: dict[str, object]) -> dict[str, object]:
    return config["static_resources"]["clusters"][0]["circuit_breakers"]["thresholds"][0]


def test_adapter_emits_auditable_policy_from_enforced_envoy_config() -> None:
    config = envoy_config()
    policy = adapt(config)
    report = audit_policy(parse_policy(json.dumps(policy).encode()))

    assert report["decision"] == "pass"
    assert policy["operation"] == {
        "id": "inference.predict",
        "method": "POST",
        "deadline_ms": 2_000,
        "idempotency_key_bound": True,
    }
    assert policy["layers"] == [
        {
            "id": "envoy/inference_listener/inference.predict",
            "max_attempts": 2,
            "per_attempt_timeout_ms": 700,
            "retry_budget_percent": 10,
            "backoff_base_ms": 50,
            "backoff_cap_ms": 200,
            "jitter": "full",
            "retry_on": ["connect_error", "connect_timeout", "reset", "unavailable"],
            "honor_retry_after": True,
        }
    ]
    canonical = json.dumps(
        config, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    assert policy["config_digest"] == hashlib.sha256(canonical).hexdigest()


def test_num_retries_becomes_total_attempts_and_fractional_seconds_are_exact() -> None:
    config = envoy_config()
    action = selected_route(config)["route"]
    action["timeout"] = "2.250s"
    action["retry_policy"]["num_retries"] = 2
    action["retry_policy"]["per_try_timeout"] = "0.600s"

    policy = adapt(config, max_amplification=3)

    assert policy["operation"]["deadline_ms"] == 2_250
    assert policy["layers"][0]["max_attempts"] == 3
    assert policy["layers"][0]["per_attempt_timeout_ms"] == 600


def test_config_object_order_does_not_change_policy_evidence() -> None:
    config = envoy_config()
    reordered = copy.deepcopy(config)
    reordered["static_resources"] = dict(reversed(list(reordered["static_resources"].items())))

    assert adapt(reordered) == adapt(config)


def test_missing_retry_override_header_fails_closed() -> None:
    config = envoy_config()
    removals = config["static_resources"]["listeners"][0]["filter_chains"][0]["filters"][0][
        "typed_config"
    ]["route_config"]["request_headers_to_remove"]
    removals.remove("x-envoy-max-retries")

    with pytest.raises(EnvoyRetryAdapterError, match="x-envoy-max-retries"):
        adapt(config)


@pytest.mark.parametrize("condition", ["5xx", "gateway-error", "retriable-4xx"])
def test_broad_or_unsafe_retry_conditions_are_rejected(condition: str) -> None:
    config = envoy_config()
    retry_policy(config)["retry_on"] = condition
    retry_policy(config).pop("retriable_status_codes")
    retry_policy(config).pop("rate_limited_retry_back_off")

    with pytest.raises(EnvoyRetryAdapterError, match="outside safe policy"):
        adapt(config)


@pytest.mark.parametrize("status", [409, 500, 502, 504])
def test_only_reviewed_status_codes_are_mapped(status: int) -> None:
    config = envoy_config()
    retry_policy(config)["retriable_status_codes"] = [status]

    with pytest.raises(EnvoyRetryAdapterError, match="status code"):
        adapt(config)


def test_dormant_status_code_list_is_rejected() -> None:
    config = envoy_config()
    retry_policy(config)["retry_on"] = "connect-failure"
    retry_policy(config).pop("rate_limited_retry_back_off")

    with pytest.raises(EnvoyRetryAdapterError, match="Dormant|dormant"):
        adapt(config)


def test_retry_after_must_be_explicit_for_unavailable_or_rate_limited() -> None:
    config = envoy_config()
    retry_policy(config)["rate_limited_retry_back_off"]["reset_headers"] = [
        {"name": "X-RateLimit-Reset", "format": "UNIX_TIMESTAMP"}
    ]

    with pytest.raises(EnvoyRetryAdapterError, match="Retry-After"):
        adapt(config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda threshold: threshold["retry_budget"].pop("min_retry_concurrency"),
            "min_retry_concurrency",
        ),
        (
            lambda threshold: threshold["retry_budget"].update(min_retry_concurrency={"value": 1}),
            "explicitly zero",
        ),
        (
            lambda threshold: threshold["retry_budget"].update(budget_interval="100ms"),
            "interval must be zero",
        ),
        (lambda threshold: threshold.update(max_retries=3), "must not coexist"),
    ],
)
def test_cluster_retry_budget_must_enforce_declared_percentage(mutation, message: str) -> None:
    config = envoy_config()
    mutation(default_threshold(config))

    with pytest.raises(EnvoyRetryAdapterError, match=message):
        adapt(config)


def test_route_requires_both_idempotency_header_and_binding_annotation() -> None:
    config = envoy_config()
    selected_route(config)["metadata"]["filter_metadata"]["retry_topology"][
        "idempotency_key_bound"
    ] = False

    policy = adapt(config)
    report = audit_policy(parse_policy(json.dumps(policy).encode()))

    assert policy["operation"]["idempotency_key_bound"] is False
    assert report["decision"] == "fail"
    assert report["violations"][0]["code"] == "unsafe_non_idempotent_retry"


def test_ambiguous_idempotency_matcher_is_rejected() -> None:
    config = envoy_config()
    headers = selected_route(config)["match"]["headers"]
    headers[1]["invert_match"] = True

    with pytest.raises(EnvoyRetryAdapterError, match="ambiguous"):
        adapt(config)


def test_dynamic_routes_and_weighted_clusters_are_outside_boundary() -> None:
    config = envoy_config()
    manager = config["static_resources"]["listeners"][0]["filter_chains"][0]["filters"][0][
        "typed_config"
    ]
    manager["rds"] = {"route_config_name": "dynamic"}

    with pytest.raises(EnvoyRetryAdapterError, match="dynamic route"):
        adapt(config)

    config = envoy_config()
    selected_route(config)["route"]["weighted_clusters"] = {"clusters": []}
    with pytest.raises(EnvoyRetryAdapterError, match="weighted"):
        adapt(config)


def test_unsupported_retry_features_fail_instead_of_being_ignored() -> None:
    config = envoy_config()
    retry_policy(config)["retry_host_predicate"] = []

    with pytest.raises(EnvoyRetryAdapterError, match="unsupported fields"):
        adapt(config)


@pytest.mark.parametrize("duration", ["", "1m", "0s", "0.0001s", True])
def test_ambiguous_or_unsafe_durations_are_rejected(duration: object) -> None:
    config = envoy_config()
    selected_route(config)["route"]["timeout"] = duration

    with pytest.raises(EnvoyRetryAdapterError, match="duration|range|milliseconds"):
        adapt(config)


def test_duplicate_selected_route_fails_closed() -> None:
    config = envoy_config()
    routes = config["static_resources"]["listeners"][0]["filter_chains"][0]["filters"][0][
        "typed_config"
    ]["route_config"]["virtual_hosts"][0]["routes"]
    routes.append(copy.deepcopy(routes[0]))

    with pytest.raises(EnvoyRetryAdapterError, match="exactly one"):
        adapt(config)


def test_cli_writes_policy_atomically_and_distinguishes_input_error(tmp_path: Path) -> None:
    source = tmp_path / "envoy.json"
    output = tmp_path / "retry-policy.json"
    source.write_text(json.dumps(envoy_config()), encoding="utf-8")
    arguments = [
        str(source),
        "--listener",
        "inference_listener",
        "--virtual-host",
        "inference_api",
        "--route",
        "inference.predict",
        "--generated-at",
        GENERATED_AT,
        "--output",
        str(output),
    ]

    assert main(arguments) == 0
    assert audit_policy(parse_policy(output.read_bytes()))["decision"] == "pass"

    source.write_text("{}", encoding="utf-8")
    assert main(arguments) == 2
    assert json.loads(output.read_text())["config_digest"]


def test_loader_rejects_duplicate_fields_symlinks_and_oversize(tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    common = [
        "--listener",
        "inference_listener",
        "--virtual-host",
        "inference_api",
        "--route",
        "inference.predict",
        "--generated-at",
        GENERATED_AT,
        "--output",
        str(output),
    ]
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"static_resources":{},"static_resources":{}}', encoding="utf-8")
    assert main([str(duplicate), *common]) == 2

    target = tmp_path / "target.json"
    target.write_text(json.dumps(envoy_config()), encoding="utf-8")
    link = tmp_path / "linked.json"
    os.symlink(target, link)
    assert main([str(link), *common]) == 2

    oversized = tmp_path / "oversized.json"
    with oversized.open("wb") as handle:
        handle.truncate(MAX_CONFIG_BYTES + 1)
    assert main([str(oversized), *common]) == 2


def test_direct_adapter_enforces_canonical_size_budget() -> None:
    config = envoy_config()
    config["padding"] = "x" * MAX_CONFIG_BYTES

    with pytest.raises(EnvoyRetryAdapterError, match="byte limit"):
        adapt(config)
