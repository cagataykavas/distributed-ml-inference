from __future__ import annotations

import json
import random
from dataclasses import asdict
from ipaddress import ip_network
from pathlib import Path

import pytest

from inference.network_acl import (
    INVALID_EVIDENCE_EXIT_CODE,
    PASS_EXIT_CODE,
    POLICY_FAILURE_EXIT_CODE,
    AclRule,
    EvidenceError,
    NetworkAcl,
    RequiredFlow,
    Segment,
    _evaluate_direction,
    audit_network_acls,
    main,
)


def rule(
    number: int,
    action: str,
    cidr: str,
    port_start: int,
    port_end: int,
) -> AclRule:
    return AclRule(number, action, "tcp", cidr, port_start, port_end)


def topology() -> tuple[list[Segment], list[NetworkAcl], list[RequiredFlow]]:
    segments = [
        Segment("alb-a", "10.42.1.0/24", "edge"),
        Segment("service-a", "10.42.11.0/24", "service"),
    ]
    acls = [
        NetworkAcl(
            "edge",
            ingress=(rule(100, "allow", "10.42.11.0/24", 32_768, 60_999),),
            egress=(rule(100, "allow", "10.42.11.0/24", 8000, 8000),),
        ),
        NetworkAcl(
            "service",
            ingress=(rule(100, "allow", "10.42.1.0/24", 8000, 8000),),
            egress=(rule(100, "allow", "10.42.1.0/24", 32_768, 60_999),),
        ),
    ]
    flows = [
        RequiredFlow(
            "alb-to-inference",
            "alb-a",
            "service-a",
            "tcp",
            8000,
            8000,
            32_768,
            60_999,
        )
    ]
    return segments, acls, flows


def payload(
    segments: list[Segment], acls: list[NetworkAcl], flows: list[RequiredFlow]
) -> dict[str, object]:
    return {
        "segments": [asdict(segment) for segment in segments],
        "acls": [asdict(acl) for acl in acls],
        "required_flows": [asdict(flow) for flow in flows],
    }


def test_complete_forward_and_ephemeral_return_path_passes() -> None:
    report = audit_network_acls(*topology())

    assert report.passed
    assert report.flow_count == 1
    assert report.flows[0].passed
    assert report.evaluated_cells == 4
    assert len(report.evidence_digest) == 64
    assert len(report.flows[0].flow_ref) == 16
    rendered = json.dumps(report.as_dict())
    assert "alb-to-inference" not in rendered
    assert "alb-a" not in rendered
    assert "service-a" not in rendered


def test_missing_return_rule_fails_closed() -> None:
    segments, acls, flows = topology()
    acls[1] = NetworkAcl("service", ingress=acls[1].ingress, egress=())

    report = audit_network_acls(segments, acls, flows)

    assert not report.passed
    assert [(item.stage, item.reason) for item in report.violations] == [
        ("return_destination_egress", "implicit_deny")
    ]


def test_earlier_deny_shadows_later_allow_for_part_of_peer_range() -> None:
    segments, acls, flows = topology()
    acls[1] = NetworkAcl(
        "service",
        ingress=(
            rule(50, "deny", "10.42.1.128/25", 8000, 8000),
            rule(100, "allow", "10.42.1.0/24", 8000, 8000),
        ),
        egress=acls[1].egress,
    )

    report = audit_network_acls(segments, acls, flows)

    violation = report.violations[0]
    assert violation.stage == "forward_destination_ingress"
    assert violation.reason == "explicit_deny"
    assert violation.blocking_rule == 50


@pytest.mark.parametrize(
    ("cidr", "port_start", "port_end"),
    [
        ("10.42.1.0/25", 8000, 8000),
        ("10.42.1.0/24", 8000, 8000),
    ],
)
def test_partial_cidr_or_port_coverage_is_rejected(
    cidr: str, port_start: int, port_end: int
) -> None:
    segments, acls, flows = topology()
    if cidr == "10.42.1.0/24":
        flows[0] = RequiredFlow(
            "alb-to-inference",
            "alb-a",
            "service-a",
            "tcp",
            8000,
            8001,
            32_768,
            60_999,
        )
    acls[1] = NetworkAcl(
        "service",
        ingress=(rule(100, "allow", cidr, port_start, port_end),),
        egress=acls[1].egress,
    )

    report = audit_network_acls(segments, acls, flows)

    assert not report.passed
    assert "forward_destination_ingress" in {violation.stage for violation in report.violations}


def test_overlapping_segments_are_invalid_evidence() -> None:
    _, acls, flows = topology()
    segments = [
        Segment("alb-a", "10.42.0.0/24", "edge"),
        Segment("service-a", "10.42.0.128/25", "service"),
    ]

    with pytest.raises(EvidenceError, match="must not overlap"):
        audit_network_acls(segments, acls, flows)


def test_input_order_does_not_change_digest_or_results() -> None:
    segments, acls, flows = topology()
    first = audit_network_acls(segments, acls, flows)
    reversed_acls = [
        NetworkAcl(acl.name, tuple(reversed(acl.ingress)), tuple(reversed(acl.egress)))
        for acl in reversed(acls)
    ]
    second = audit_network_acls(list(reversed(segments)), reversed_acls, flows)

    assert first.evidence_digest == second.evidence_digest
    assert first.flows == second.flows


def test_ipv6_ranges_are_evaluated_without_ipv4_coercion() -> None:
    segments = [
        Segment("edge-v6", "2001:db8:1::/64", "edge"),
        Segment("service-v6", "2001:db8:2::/64", "service"),
    ]
    acls = [
        NetworkAcl(
            "edge",
            ingress=(rule(100, "allow", "2001:db8:2::/64", 32_768, 60_999),),
            egress=(rule(100, "allow", "2001:db8:2::/64", 8000, 8000),),
        ),
        NetworkAcl(
            "service",
            ingress=(rule(100, "allow", "2001:db8:1::/64", 8000, 8000),),
            egress=(rule(100, "allow", "2001:db8:1::/64", 32_768, 60_999),),
        ),
    ]
    flows = [
        RequiredFlow(
            "ipv6-inference",
            "edge-v6",
            "service-v6",
            "tcp",
            8000,
            8000,
            32_768,
            60_999,
        )
    ]

    assert audit_network_acls(segments, acls, flows).passed


def test_partitioned_evaluator_matches_exhaustive_small_domain() -> None:
    generator = random.Random(17)
    peer = ip_network("192.0.2.0/30")
    cidrs = ("192.0.2.0/30", "192.0.2.0/31", "192.0.2.2/31")
    for _ in range(200):
        rules = tuple(
            rule(
                number,
                generator.choice(("allow", "deny")),
                generator.choice(cidrs),
                port_start := generator.randint(1000, 1003),
                generator.randint(port_start, 1003),
            )
            for number in range(10, 60, 10)
        )
        expected = True
        for address in peer:
            for port in range(1000, 1004):
                decision = next(
                    (
                        candidate.action
                        for candidate in sorted(rules, key=lambda item: item.number)
                        if address in candidate.network
                        and candidate.port_start <= port <= candidate.port_end
                    ),
                    "deny",
                )
                expected &= decision == "allow"
        actual, _, _, _, _ = _evaluate_direction(
            rules,
            peer=peer,
            protocol="tcp",
            port_start=1000,
            port_end=1003,
        )
        assert actual is expected


def test_cli_distinguishes_pass_policy_failure_and_invalid_evidence(tmp_path: Path) -> None:
    segments, acls, flows = topology()
    passing = tmp_path / "passing.json"
    failing = tmp_path / "failing.json"
    invalid = tmp_path / "invalid.json"
    output = tmp_path / "report.json"
    passing.write_text(json.dumps(payload(segments, acls, flows)), encoding="utf-8")
    failing_acls = [acls[0], NetworkAcl("service", acls[1].ingress, ())]
    failing.write_text(json.dumps(payload(segments, failing_acls, flows)), encoding="utf-8")
    invalid.write_text('{"segments": [], "segments": []}', encoding="utf-8")

    assert main([str(passing), "--output", str(output)]) == PASS_EXIT_CODE
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
    assert main([str(failing)]) == POLICY_FAILURE_EXIT_CODE
    assert main([str(invalid)]) == INVALID_EVIDENCE_EXIT_CODE
