from __future__ import annotations

import copy
import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from envoy_retry_adapter import adapt_envoy_retry_config
from envoy_runtime_attestation import (
    RuntimeAttestationError,
    attest_runtime_config,
    main,
    sign_runtime_envelope,
)
from retry_topology_audit import parse_policy
from tests.test_envoy_retry_adapter import envoy_config

CAPTURED_AT = "2026-11-09T20:00:00Z"
NOW = datetime.fromisoformat("2026-11-09T20:00:30+00:00")
NONCE = "release-check-9f3a7c2d"
KEY = bytes(range(32))


def policy_dict(config=None):
    return adapt_envoy_retry_config(
        envoy_config() if config is None else config,
        listener_name="inference_listener",
        virtual_host_name="inference_api",
        route_name="inference.predict",
        generated_at="2026-11-09T19:55:00Z",
        max_amplification=2,
        max_retry_layers=1,
        max_retry_budget_percent=20,
    )


def config_dump(config=None):
    return {
        "configs": [
            {
                "@type": "type.googleapis.com/envoy.admin.v3.BootstrapConfigDump",
                "bootstrap": envoy_config() if config is None else config,
            }
        ]
    }


def envelope(config=None, **overrides):
    fields = {
        "schema_version": 1,
        "captured_at": CAPTURED_AT,
        "nonce": NONCE,
        "config_dump": config_dump(config),
    }
    fields.update(overrides)
    return sign_runtime_envelope(fields, KEY)


def attest(value=None, **overrides):
    arguments = {
        "expected_nonce": NONCE,
        "now": NOW,
        "max_age_seconds": 60,
        "listener_name": "inference_listener",
        "virtual_host_name": "inference_api",
        "route_name": "inference.predict",
    }
    arguments.update(overrides)
    policy = parse_policy(json.dumps(policy_dict(), separators=(",", ":")).encode())
    return attest_runtime_config(policy, envelope() if value is None else value, KEY, **arguments)


def test_authenticated_matching_runtime_policy_passes() -> None:
    report = attest()

    assert report["decision"] == "pass"
    assert report["authenticated"] is True
    assert report["violations"] == []
    assert report["reviewed_config_digest"] == policy_dict()["config_digest"]
    assert report["runtime_dump_digest"] != report["runtime_bootstrap_digest"]


def test_authenticated_retry_budget_drift_is_a_policy_failure() -> None:
    runtime = envoy_config()
    runtime["static_resources"]["clusters"][0]["circuit_breakers"]["thresholds"][0]["retry_budget"][
        "budget_percent"
    ]["value"] = 15

    report = attest(envelope(runtime))

    assert report["decision"] == "fail"
    assert report["violations"] == ["retry_layers_drift"]


def test_authenticated_operation_drift_is_a_policy_failure() -> None:
    runtime = envoy_config()
    runtime["static_resources"]["listeners"][0]["filter_chains"][0]["filters"][0]["typed_config"][
        "route_config"
    ]["virtual_hosts"][0]["routes"][0]["match"]["headers"][0]["exact_match"] = "PUT"

    report = attest(envelope(runtime))

    assert report["decision"] == "fail"
    assert report["violations"] == ["operation_drift"]


def test_tampered_dump_fails_authentication_before_comparison() -> None:
    signed = envelope()
    signed["config_dump"]["configs"][0]["bootstrap"]["static_resources"]["clusters"][0]["name"] = (
        "tampered"
    )

    with pytest.raises(RuntimeAttestationError, match="authentication failed"):
        attest(signed)


def test_wrong_key_and_nonce_are_rejected() -> None:
    policy = parse_policy(json.dumps(policy_dict()).encode())
    with pytest.raises(RuntimeAttestationError, match="authentication failed"):
        attest_runtime_config(
            policy,
            envelope(),
            b"x" * 32,
            expected_nonce=NONCE,
            now=NOW,
            max_age_seconds=60,
            listener_name="inference_listener",
            virtual_host_name="inference_api",
            route_name="inference.predict",
        )
    with pytest.raises(RuntimeAttestationError, match="challenge"):
        attest(expected_nonce="another-release-check")


@pytest.mark.parametrize(
    ("captured_at", "message"),
    [
        ("2026-11-09T19:58:00Z", "stale"),
        ("2026-11-09T20:00:36Z", "future"),
        ("2026-11-09T20:00:00", "explicit UTC offset"),
    ],
)
def test_freshness_window_rejects_stale_future_and_naive_evidence(
    captured_at: str, message: str
) -> None:
    with pytest.raises(RuntimeAttestationError, match=message):
        attest(envelope(captured_at=captured_at))


def test_bootstrap_snapshot_must_be_unique_and_bounded() -> None:
    duplicate = config_dump()
    duplicate["configs"].append(copy.deepcopy(duplicate["configs"][0]))

    with pytest.raises(RuntimeAttestationError, match="exactly one"):
        attest(envelope(config_dump=duplicate))
    with pytest.raises(RuntimeAttestationError, match="bounded non-empty"):
        attest(envelope(config_dump={"configs": []}))


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    policy_path = tmp_path / "policy.json"
    envelope_path = tmp_path / "runtime.json"
    key_path = tmp_path / "attestation.key"
    policy_path.write_text(json.dumps(policy_dict()), encoding="utf-8")
    envelope_path.write_text(json.dumps(envelope()), encoding="utf-8")
    key_path.write_bytes(KEY)
    key_path.chmod(0o600)
    return policy_path, envelope_path, key_path


def cli_arguments(tmp_path: Path) -> list[str]:
    policy_path, envelope_path, key_path = write_inputs(tmp_path)
    return [
        str(policy_path),
        str(envelope_path),
        "--key-file",
        str(key_path),
        "--expected-nonce",
        NONCE,
        "--listener",
        "inference_listener",
        "--virtual-host",
        "inference_api",
        "--route",
        "inference.predict",
        "--now",
        "2026-11-09T20:00:30Z",
    ]


def test_cli_writes_atomic_attestation_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    assert main([*cli_arguments(tmp_path), "--output", str(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["decision"] == "pass"
    assert not list(tmp_path.glob(".report.json.*"))


def test_cli_rejects_exposed_key_and_duplicate_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = cli_arguments(tmp_path)
    key_path = Path(arguments[arguments.index("--key-file") + 1])
    key_path.chmod(0o644)

    assert main(arguments) == 2
    assert "group/world" in capsys.readouterr().err

    key_path.chmod(0o600)
    envelope_path = Path(arguments[1])
    envelope_path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    assert main(arguments) == 2
    assert "duplicate JSON field" in capsys.readouterr().err


def test_key_loader_rejects_symlinks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks unavailable")
    arguments = cli_arguments(tmp_path)
    original = Path(arguments[arguments.index("--key-file") + 1])
    link = tmp_path / "linked.key"
    link.symlink_to(original)
    arguments[arguments.index("--key-file") + 1] = str(link)

    assert main(arguments) == 2
    assert "unavailable" in capsys.readouterr().err
