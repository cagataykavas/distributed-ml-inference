"""Normalize a bounded Envoy static route into retry-topology policy evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from retry_topology_audit import PolicyInputError, parse_policy

MAX_CONFIG_BYTES = 2 * 1024 * 1024
MAX_COLLECTION_ITEMS = 1_024
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")
_DURATION = re.compile(
    r"^(?:(?P<milliseconds>[0-9]+)ms|(?P<seconds>[0-9]+)(?:\.(?P<fraction>[0-9]{1,3}))?s)$"
)
_HCM_TYPE = "type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager"
_OVERRIDE_HEADERS = frozenset(
    {
        "x-envoy-hedge-on-per-try-timeout",
        "x-envoy-max-retries",
        "x-envoy-retriable-header-names",
        "x-envoy-retriable-status-codes",
        "x-envoy-retry-grpc-on",
        "x-envoy-retry-on",
        "x-envoy-upstream-rq-per-try-idle-timeout-ms",
        "x-envoy-upstream-rq-per-try-timeout-ms",
        "x-envoy-upstream-rq-timeout-ms",
    }
)
_SUPPORTED_RETRY_FIELDS = frozenset(
    {
        "num_retries",
        "per_try_timeout",
        "rate_limited_retry_back_off",
        "retriable_status_codes",
        "retry_back_off",
        "retry_on",
    }
)
_SAFE_HTTP_METHODS = frozenset({"DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"})


class EnvoyRetryAdapterError(ValueError):
    """Envoy configuration cannot be normalized without guessing."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise EnvoyRetryAdapterError("Envoy JSON contains duplicate fields")
        value[key] = item
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EnvoyRetryAdapterError("Envoy configuration is not canonical JSON") from exc


def _load_json(path: Path) -> object:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EnvoyRetryAdapterError("Envoy config is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EnvoyRetryAdapterError("Envoy config must be a regular file")
        if not 1 <= before.st_size <= MAX_CONFIG_BYTES:
            raise EnvoyRetryAdapterError("Envoy config exceeds its byte limit")
        chunks: list[bytes] = []
        remaining = MAX_CONFIG_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if len(payload) != before.st_size or len(payload) > MAX_CONFIG_BYTES:
            raise EnvoyRetryAdapterError("Envoy config changed while being read")
        if identity_before != identity_after:
            raise EnvoyRetryAdapterError("Envoy config changed while being read")
    finally:
        os.close(descriptor)
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                EnvoyRetryAdapterError("Envoy JSON contains a non-finite number")
            ),
        )
    except EnvoyRetryAdapterError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvoyRetryAdapterError("Envoy config is not valid UTF-8 JSON") from exc


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EnvoyRetryAdapterError(f"{context} must be an object")
    return value


def _array(value: object, context: str) -> list[Any]:
    if not isinstance(value, list) or len(value) > MAX_COLLECTION_ITEMS:
        raise EnvoyRetryAdapterError(f"{context} must be a bounded array")
    return value


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise EnvoyRetryAdapterError(f"{context} must be a bounded identifier")
    return value


def _integer(value: object, context: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise EnvoyRetryAdapterError(f"{context} must be an integer in [{minimum}, {maximum}]")
    return value


def _wrapped_integer(value: object, context: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, dict):
        if set(value) != {"value"}:
            raise EnvoyRetryAdapterError(f"{context} wrapper has unexpected fields")
        value = value["value"]
    return _integer(value, context, minimum=minimum, maximum=maximum)


def _duration_ms(value: object, context: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, str):
        raise EnvoyRetryAdapterError(f"{context} must be an explicit duration")
    matched = _DURATION.fullmatch(value)
    if matched is None:
        raise EnvoyRetryAdapterError(f"{context} must use integer milliseconds or seconds")
    if matched.group("milliseconds") is not None:
        milliseconds = int(matched.group("milliseconds"))
    else:
        milliseconds = int(matched.group("seconds")) * 1_000
        fraction = matched.group("fraction") or ""
        milliseconds += int(fraction.ljust(3, "0")) if fraction else 0
    minimum = 0 if allow_zero else 1
    if not minimum <= milliseconds <= 300_000:
        raise EnvoyRetryAdapterError(f"{context} is outside the supported duration range")
    return milliseconds


def _exactly_one(items: list[dict[str, Any]], *, name: str, context: str) -> dict[str, Any]:
    matches = [item for item in items if item.get("name") == name]
    if len(matches) != 1:
        raise EnvoyRetryAdapterError(f"{context} must contain exactly one selected {name!r}")
    return matches[0]


def _selected_route(
    config: dict[str, Any],
    *,
    listener_name: str,
    virtual_host_name: str,
    route_name: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    static = _mapping(config.get("static_resources"), "static_resources")
    listeners = [
        _mapping(item, "listener") for item in _array(static.get("listeners"), "listeners")
    ]
    listener = _exactly_one(listeners, name=listener_name, context="listeners")
    managers: list[dict[str, Any]] = []
    for chain_value in _array(listener.get("filter_chains"), "listener.filter_chains"):
        chain = _mapping(chain_value, "filter_chain")
        for filter_value in _array(chain.get("filters"), "filter_chain.filters"):
            network_filter = _mapping(filter_value, "network filter")
            if network_filter.get("name") != "envoy.filters.network.http_connection_manager":
                continue
            typed = _mapping(network_filter.get("typed_config"), "http_connection_manager")
            if typed.get("@type") != _HCM_TYPE:
                raise EnvoyRetryAdapterError("HTTP connection manager type URL is unsupported")
            managers.append(typed)
    if len(managers) != 1:
        raise EnvoyRetryAdapterError("selected listener must contain exactly one HTTP manager")
    manager = managers[0]
    if "rds" in manager or "scoped_routes" in manager:
        raise EnvoyRetryAdapterError("dynamic route discovery is outside this adapter boundary")
    route_config = _mapping(manager.get("route_config"), "route_config")
    virtual_hosts = [
        _mapping(item, "virtual_host")
        for item in _array(route_config.get("virtual_hosts"), "virtual_hosts")
    ]
    virtual_host = _exactly_one(virtual_hosts, name=virtual_host_name, context="virtual_hosts")
    routes = [_mapping(item, "route") for item in _array(virtual_host.get("routes"), "routes")]
    route = _exactly_one(routes, name=route_name, context="routes")
    return static, route_config, virtual_host, route


def _header_removals(*levels: dict[str, Any]) -> set[str]:
    removed: set[str] = set()
    for index, level in enumerate(levels):
        values = level.get("request_headers_to_remove", [])
        for value in _array(values, f"request_headers_to_remove[{index}]"):
            if not isinstance(value, str) or not value:
                raise EnvoyRetryAdapterError("request header removals must be strings")
            removed.add(value.lower())
    missing = sorted(_OVERRIDE_HEADERS - removed)
    if missing:
        raise EnvoyRetryAdapterError(
            f"retry override headers are not removed: {', '.join(missing)}"
        )
    return removed


def _operation(route: dict[str, Any]) -> tuple[str, bool]:
    match = _mapping(route.get("match"), "route.match")
    headers = [
        _mapping(item, "route header matcher")
        for item in _array(match.get("headers", []), "route.match.headers")
    ]
    method_matchers = [item for item in headers if item.get("name") == ":method"]
    if len(method_matchers) != 1 or set(method_matchers[0]) != {"name", "exact_match"}:
        raise EnvoyRetryAdapterError("route must exactly match one HTTP method")
    method = method_matchers[0]["exact_match"]
    if not isinstance(method, str) or method.upper() not in _SAFE_HTTP_METHODS:
        raise EnvoyRetryAdapterError("route HTTP method is unsupported")

    key_matchers = [
        item for item in headers if str(item.get("name", "")).lower() == "x-idempotency-key"
    ]
    if len(key_matchers) > 1:
        raise EnvoyRetryAdapterError("idempotency header matcher must be unique")
    if key_matchers and set(key_matchers[0]) != {"name", "present_match"}:
        raise EnvoyRetryAdapterError("idempotency header matcher is ambiguous")
    key_required = bool(key_matchers) and key_matchers[0].get("present_match") is True
    metadata = _mapping(route.get("metadata", {}), "route.metadata")
    filter_metadata = _mapping(metadata.get("filter_metadata", {}), "route.filter_metadata")
    annotation = _mapping(filter_metadata.get("retry_topology", {}), "retry_topology metadata")
    if set(annotation) - {"idempotency_key_bound"}:
        raise EnvoyRetryAdapterError("retry topology metadata has unknown fields")
    annotated = annotation.get("idempotency_key_bound") is True
    if (
        "idempotency_key_bound" in annotation
        and type(annotation["idempotency_key_bound"]) is not bool
    ):
        raise EnvoyRetryAdapterError("idempotency binding annotation must be boolean")
    return method.upper(), key_required and annotated


def _retry_conditions(retry_policy: dict[str, Any]) -> tuple[list[str], bool]:
    retry_on = retry_policy.get("retry_on")
    if not isinstance(retry_on, str):
        raise EnvoyRetryAdapterError("retry_policy.retry_on must be explicit")
    tokens = [token.strip() for token in retry_on.split(",")]
    if (
        not tokens
        or len(tokens) > 16
        or any(not token for token in tokens)
        or len(tokens) != len(set(tokens))
    ):
        raise EnvoyRetryAdapterError("retry_policy.retry_on is empty or ambiguous")
    conditions: set[str] = set()
    status_codes = _array(retry_policy.get("retriable_status_codes", []), "retriable_status_codes")
    if any(type(code) is not int for code in status_codes):
        raise EnvoyRetryAdapterError("retriable_status_codes must be an integer list")
    if len(status_codes) != len(set(status_codes)):
        raise EnvoyRetryAdapterError("retriable_status_codes contains duplicates")
    for token in tokens:
        if token == "connect-failure":
            conditions.update({"connect_error", "connect_timeout"})
        elif token in {"refused-stream", "reset", "reset-before-request"}:
            conditions.add("reset")
        elif token == "envoy-ratelimited":
            conditions.add("rate_limited")
        elif token == "retriable-status-codes":
            if not status_codes:
                raise EnvoyRetryAdapterError("retriable-status-codes requires explicit codes")
            for code in status_codes:
                if code == 429:
                    conditions.add("rate_limited")
                elif code == 503:
                    conditions.add("unavailable")
                else:
                    raise EnvoyRetryAdapterError("retriable status code is outside safe policy")
        else:
            raise EnvoyRetryAdapterError(f"retry condition {token!r} is outside safe policy")
    if status_codes and "retriable-status-codes" not in tokens:
        raise EnvoyRetryAdapterError("dormant retriable_status_codes are not allowed")

    needs_server_delay = bool({"rate_limited", "unavailable"}.intersection(conditions))
    rate_backoff = retry_policy.get("rate_limited_retry_back_off")
    if not needs_server_delay:
        if rate_backoff is not None:
            raise EnvoyRetryAdapterError("dormant rate-limited backoff is not allowed")
        return sorted(conditions), False
    backoff = _mapping(rate_backoff, "rate_limited_retry_back_off")
    if set(backoff) != {"reset_headers", "max_interval"}:
        raise EnvoyRetryAdapterError("rate-limited backoff fields are incomplete")
    _duration_ms(backoff["max_interval"], "rate_limited_retry_back_off.max_interval")
    headers = [
        _mapping(item, "reset header") for item in _array(backoff["reset_headers"], "reset_headers")
    ]
    honors_retry_after = any(
        set(header) == {"name", "format"}
        and str(header["name"]).lower() == "retry-after"
        and header["format"] == "SECONDS"
        for header in headers
    )
    if not honors_retry_after:
        raise EnvoyRetryAdapterError("Retry-After seconds backoff must be configured")
    return sorted(conditions), True


def _retry_budget(static: dict[str, Any], cluster_name: str) -> int:
    clusters = [_mapping(item, "cluster") for item in _array(static.get("clusters"), "clusters")]
    cluster = _exactly_one(clusters, name=cluster_name, context="clusters")
    breakers = _mapping(cluster.get("circuit_breakers"), "cluster.circuit_breakers")
    thresholds = [
        _mapping(item, "circuit breaker threshold")
        for item in _array(breakers.get("thresholds"), "circuit_breaker.thresholds")
    ]
    defaults = [item for item in thresholds if item.get("priority", "DEFAULT") == "DEFAULT"]
    if len(defaults) != 1:
        raise EnvoyRetryAdapterError("cluster must have exactly one DEFAULT threshold")
    threshold = defaults[0]
    if "max_retries" in threshold:
        raise EnvoyRetryAdapterError("retry_budget must not coexist with max_retries")
    budget = _mapping(threshold.get("retry_budget"), "retry_budget")
    if set(budget) - {"budget_percent", "budget_interval", "min_retry_concurrency"}:
        raise EnvoyRetryAdapterError("retry_budget has unsupported fields")
    percent = _mapping(budget.get("budget_percent"), "retry_budget.budget_percent")
    if set(percent) != {"value"}:
        raise EnvoyRetryAdapterError("retry budget percent must contain only value")
    budget_percent = _integer(
        percent["value"], "retry_budget.budget_percent.value", minimum=1, maximum=100
    )
    minimum = _wrapped_integer(
        budget.get("min_retry_concurrency"),
        "retry_budget.min_retry_concurrency",
        minimum=0,
        maximum=1_000_000,
    )
    if minimum != 0:
        raise EnvoyRetryAdapterError("min_retry_concurrency must be explicitly zero")
    if (
        _duration_ms(
            budget.get("budget_interval", "0s"),
            "retry_budget.budget_interval",
            allow_zero=True,
        )
        != 0
    ):
        raise EnvoyRetryAdapterError("retry budget interval must be zero")
    return budget_percent


def adapt_envoy_retry_config(
    config: object,
    *,
    listener_name: str,
    virtual_host_name: str,
    route_name: str,
    generated_at: str,
    max_amplification: int,
    max_retry_layers: int,
    max_retry_budget_percent: int,
) -> dict[str, Any]:
    root = _mapping(config, "Envoy config")
    listener = _identifier(listener_name, "listener name")
    virtual_host = _identifier(virtual_host_name, "virtual host name")
    selected_route = _identifier(route_name, "route name")
    try:
        parsed_time = datetime.fromisoformat(generated_at)
    except (TypeError, ValueError) as exc:
        raise EnvoyRetryAdapterError("generated_at must be an RFC3339 timestamp") from exc
    if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
        raise EnvoyRetryAdapterError("generated_at must include an explicit UTC offset")
    limits = {
        "max_amplification": _integer(
            max_amplification, "max_amplification", minimum=1, maximum=1_000
        ),
        "max_retry_layers": _integer(max_retry_layers, "max_retry_layers", minimum=0, maximum=16),
        "max_retry_budget_percent": _integer(
            max_retry_budget_percent,
            "max_retry_budget_percent",
            minimum=0,
            maximum=100,
        ),
    }
    static, route_config, host, route = _selected_route(
        root,
        listener_name=listener,
        virtual_host_name=virtual_host,
        route_name=selected_route,
    )
    _header_removals(route_config, host, route)
    method, idempotency_key_bound = _operation(route)
    action = _mapping(route.get("route"), "route action")
    cluster_name = _identifier(action.get("cluster"), "route cluster")
    if any(field in action for field in ("cluster_header", "weighted_clusters")):
        raise EnvoyRetryAdapterError("dynamic or weighted clusters are unsupported")
    deadline_ms = _duration_ms(action.get("timeout"), "route timeout")
    retry_policy = _mapping(action.get("retry_policy"), "route retry_policy")
    unsupported = sorted(set(retry_policy) - _SUPPORTED_RETRY_FIELDS)
    if unsupported:
        raise EnvoyRetryAdapterError(
            f"retry_policy has unsupported fields: {', '.join(unsupported)}"
        )
    retries = _wrapped_integer(
        retry_policy.get("num_retries"),
        "retry_policy.num_retries",
        minimum=1,
        maximum=9,
    )
    per_try_timeout_ms = _duration_ms(
        retry_policy.get("per_try_timeout"), "retry_policy.per_try_timeout"
    )
    retry_backoff = _mapping(retry_policy.get("retry_back_off"), "retry_back_off")
    if set(retry_backoff) != {"base_interval", "max_interval"}:
        raise EnvoyRetryAdapterError("retry_back_off must explicitly set base and maximum")
    backoff_base_ms = _duration_ms(retry_backoff["base_interval"], "retry_back_off.base_interval")
    backoff_cap_ms = _duration_ms(retry_backoff["max_interval"], "retry_back_off.max_interval")
    if backoff_cap_ms < backoff_base_ms:
        raise EnvoyRetryAdapterError("retry backoff maximum is below its base")
    conditions, honors_retry_after = _retry_conditions(retry_policy)
    retry_budget_percent = _retry_budget(static, cluster_name)
    canonical_config = _canonical_json(root)
    if len(canonical_config) > MAX_CONFIG_BYTES:
        raise EnvoyRetryAdapterError("Envoy config exceeds its byte limit")
    policy = {
        "schema_version": 1,
        "generated_at": generated_at,
        "config_digest": hashlib.sha256(canonical_config).hexdigest(),
        "operation": {
            "id": selected_route,
            "method": method,
            "deadline_ms": deadline_ms,
            "idempotency_key_bound": idempotency_key_bound,
        },
        "limits": limits,
        "layers": [
            {
                "id": f"envoy/{listener}/{selected_route}",
                "max_attempts": retries + 1,
                "per_attempt_timeout_ms": per_try_timeout_ms,
                "retry_budget_percent": retry_budget_percent,
                "backoff_base_ms": backoff_base_ms,
                "backoff_cap_ms": backoff_cap_ms,
                "jitter": "full",
                "retry_on": conditions,
                "honor_retry_after": honors_retry_after,
            }
        ],
    }
    try:
        parse_policy(_canonical_json(policy))
    except PolicyInputError as exc:  # pragma: no cover - adapter invariant
        raise EnvoyRetryAdapterError("normalized policy violates the audit schema") from exc
    return policy


def _write_json(value: object, output: Path | None) -> None:
    payload = _canonical_json(value) + b"\n"
    if output is None:
        sys.stdout.write(payload.decode("utf-8"))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize a static Envoy route into retry-topology evidence"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--listener", required=True)
    parser.add_argument("--virtual-host", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--max-amplification", type=int, default=2)
    parser.add_argument("--max-retry-layers", type=int, default=1)
    parser.add_argument("--max-retry-budget-percent", type=int, default=20)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        policy = adapt_envoy_retry_config(
            _load_json(arguments.config),
            listener_name=arguments.listener,
            virtual_host_name=arguments.virtual_host,
            route_name=arguments.route,
            generated_at=arguments.generated_at,
            max_amplification=arguments.max_amplification,
            max_retry_layers=arguments.max_retry_layers,
            max_retry_budget_percent=arguments.max_retry_budget_percent,
        )
        _write_json(policy, arguments.output)
    except (EnvoyRetryAdapterError, OSError) as exc:
        sys.stderr.write(f"Envoy retry config error: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
