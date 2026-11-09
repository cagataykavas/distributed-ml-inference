"""Authenticate an Envoy config dump and compare its retry policy with reviewed evidence."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import stat
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from envoy_retry_adapter import EnvoyRetryAdapterError, adapt_envoy_retry_config
from retry_topology_audit import MAX_POLICY_BYTES, PolicyInputError, RetryPolicy, parse_policy

MAX_ENVELOPE_BYTES = 2 * 1024 * 1024
MAX_CONFIG_DUMP_ITEMS = 128
MAX_CLOCK_SKEW_SECONDS = 5
_NONCE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BOOTSTRAP_TYPE = "type.googleapis.com/envoy.admin.v3.BootstrapConfigDump"


class RuntimeAttestationError(ValueError):
    """Runtime evidence is malformed, stale, or unauthenticated."""


def _duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeAttestationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


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
        raise RuntimeAttestationError("runtime evidence is not canonical JSON") from exc


def _read_regular_file(path: Path, *, maximum: int, secret: bool = False) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeAttestationError(f"{path.name} is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeAttestationError(f"{path.name} must be a singly linked regular file")
        if secret and stat.S_IMODE(before.st_mode) & 0o077:
            raise RuntimeAttestationError("attestation key must not be group/world accessible")
        if not 1 <= before.st_size <= maximum:
            raise RuntimeAttestationError(f"{path.name} exceeds its byte limit")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        identity = lambda item: (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if len(payload) != before.st_size or identity(before) != identity(after):
            raise RuntimeAttestationError(f"{path.name} changed while being read")
        return payload
    finally:
        os.close(descriptor)


def _parse_json(payload: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_keys,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                RuntimeAttestationError(f"{context} contains a non-finite number")
            ),
        )
    except RuntimeAttestationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeAttestationError(f"{context} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeAttestationError(f"{context} must be an object")
    return value


def _timestamp(value: object, context: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeAttestationError(f"{context} must be an RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RuntimeAttestationError(f"{context} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeAttestationError(f"{context} must include an explicit UTC offset")
    return parsed


def _signed_fields(envelope: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": envelope["schema_version"],
        "captured_at": envelope["captured_at"],
        "nonce": envelope["nonce"],
        "config_dump": envelope["config_dump"],
    }


def sign_runtime_envelope(fields: dict[str, Any], key: bytes) -> dict[str, Any]:
    """Return an HMAC-SHA256 envelope; intended for collectors and deterministic tests."""
    if not 32 <= len(key) <= 64:
        raise RuntimeAttestationError("attestation key must contain 32 to 64 bytes")
    if set(fields) != {"schema_version", "captured_at", "nonce", "config_dump"}:
        raise RuntimeAttestationError("signed runtime fields are incomplete or unknown")
    envelope = dict(fields)
    envelope["hmac_sha256"] = hmac.new(key, _canonical_json(fields), hashlib.sha256).hexdigest()
    return envelope


def _bootstrap_from_dump(config_dump: object) -> dict[str, Any]:
    if not isinstance(config_dump, dict) or set(config_dump) != {"configs"}:
        raise RuntimeAttestationError("config_dump must contain only configs")
    configs = config_dump["configs"]
    if not isinstance(configs, list) or not 1 <= len(configs) <= MAX_CONFIG_DUMP_ITEMS:
        raise RuntimeAttestationError("config_dump.configs must be a bounded non-empty array")
    bootstrap_dumps = []
    for item in configs:
        if not isinstance(item, dict):
            raise RuntimeAttestationError("config_dump entries must be objects")
        if item.get("@type") == _BOOTSTRAP_TYPE:
            bootstrap_dumps.append(item)
    if len(bootstrap_dumps) != 1:
        raise RuntimeAttestationError("config_dump must contain exactly one bootstrap snapshot")
    selected = bootstrap_dumps[0]
    if set(selected) != {"@type", "bootstrap"} or not isinstance(selected["bootstrap"], dict):
        raise RuntimeAttestationError("bootstrap snapshot has unsupported or missing fields")
    return selected["bootstrap"]


def attest_runtime_config(
    reviewed_policy: RetryPolicy,
    envelope: dict[str, Any],
    key: bytes,
    *,
    expected_nonce: str,
    now: datetime,
    max_age_seconds: int,
    listener_name: str,
    virtual_host_name: str,
    route_name: str,
) -> dict[str, Any]:
    """Verify provenance/freshness and report retry-policy equivalence."""
    expected_fields = {"schema_version", "captured_at", "nonce", "config_dump", "hmac_sha256"}
    if set(envelope) != expected_fields or envelope.get("schema_version") != 1:
        raise RuntimeAttestationError("runtime envelope schema is unsupported")
    nonce = envelope["nonce"]
    if not isinstance(nonce, str) or _NONCE.fullmatch(nonce) is None:
        raise RuntimeAttestationError("runtime nonce must be 16 to 128 URL-safe characters")
    if nonce != expected_nonce:
        raise RuntimeAttestationError("runtime nonce does not match the verifier challenge")
    signature = envelope["hmac_sha256"]
    if not isinstance(signature, str) or _SHA256.fullmatch(signature) is None:
        raise RuntimeAttestationError("runtime HMAC must be lowercase SHA-256 hex")
    if not 32 <= len(key) <= 64:
        raise RuntimeAttestationError("attestation key must contain 32 to 64 bytes")
    expected_signature = hmac.new(
        key, _canonical_json(_signed_fields(envelope)), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise RuntimeAttestationError("runtime envelope authentication failed")

    if now.tzinfo is None or now.utcoffset() is None:
        raise RuntimeAttestationError("verification time must include an explicit UTC offset")
    if type(max_age_seconds) is not int or not 1 <= max_age_seconds <= 3_600:
        raise RuntimeAttestationError("max age must be between 1 and 3600 seconds")
    captured = _timestamp(envelope["captured_at"], "captured_at")
    age = (now.astimezone(UTC) - captured.astimezone(UTC)).total_seconds()
    if age < -MAX_CLOCK_SKEW_SECONDS:
        raise RuntimeAttestationError("runtime evidence is dated in the future")
    if age > max_age_seconds:
        raise RuntimeAttestationError("runtime evidence is stale")

    bootstrap = _bootstrap_from_dump(envelope["config_dump"])
    try:
        runtime_data = adapt_envoy_retry_config(
            bootstrap,
            listener_name=listener_name,
            virtual_host_name=virtual_host_name,
            route_name=route_name,
            generated_at=envelope["captured_at"],
            max_amplification=reviewed_policy.limits.max_amplification,
            max_retry_layers=reviewed_policy.limits.max_retry_layers,
            max_retry_budget_percent=reviewed_policy.limits.max_retry_budget_percent,
        )
        runtime_policy = parse_policy(_canonical_json(runtime_data))
    except (EnvoyRetryAdapterError, PolicyInputError) as exc:
        raise RuntimeAttestationError("runtime bootstrap cannot be safely normalized") from exc

    comparisons = {
        "operation_drift": reviewed_policy.operation != runtime_policy.operation,
        "retry_layers_drift": reviewed_policy.layers != runtime_policy.layers,
    }
    violations = [code for code, differs in comparisons.items() if differs]
    return {
        "schema_version": 1,
        "decision": "pass" if not violations else "fail",
        "captured_at": envelope["captured_at"],
        "nonce": nonce,
        "operation_id": reviewed_policy.operation.operation_id,
        "reviewed_config_digest": reviewed_policy.config_digest,
        "runtime_bootstrap_digest": runtime_policy.config_digest,
        "runtime_dump_digest": hashlib.sha256(_canonical_json(envelope["config_dump"])).hexdigest(),
        "authenticated": True,
        "violations": violations,
    }


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
        description="Authenticate an Envoy config dump and detect reviewed retry-policy drift"
    )
    parser.add_argument("policy", type=Path)
    parser.add_argument("envelope", type=Path)
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--expected-nonce", required=True)
    parser.add_argument("--listener", required=True)
    parser.add_argument("--virtual-host", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--max-age-seconds", type=int, default=60)
    parser.add_argument("--now", help="verification timestamp; defaults to current UTC time")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        policy = parse_policy(_read_regular_file(arguments.policy, maximum=MAX_POLICY_BYTES))
        envelope = _parse_json(
            _read_regular_file(arguments.envelope, maximum=MAX_ENVELOPE_BYTES),
            "runtime envelope",
        )
        key = _read_regular_file(arguments.key_file, maximum=64, secret=True)
        now = _timestamp(arguments.now, "now") if arguments.now is not None else datetime.now(UTC)
        report = attest_runtime_config(
            policy,
            envelope,
            key,
            expected_nonce=arguments.expected_nonce,
            now=now,
            max_age_seconds=arguments.max_age_seconds,
            listener_name=arguments.listener,
            virtual_host_name=arguments.virtual_host,
            route_name=arguments.route,
        )
        _write_json(report, arguments.output)
    except (OSError, PolicyInputError, RuntimeAttestationError) as exc:
        sys.stderr.write(f"runtime attestation input error: {exc}\n")
        return 2
    return 0 if report["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
