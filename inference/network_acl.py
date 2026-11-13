from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PASS_EXIT_CODE = 0
POLICY_FAILURE_EXIT_CODE = 2
INVALID_EVIDENCE_EXIT_CODE = 3
MAX_INPUT_BYTES = 2_000_000
MAX_SEGMENTS = 64
MAX_ACLS = 64
MAX_RULES_PER_DIRECTION = 64
MAX_FLOWS = 32
MAX_EVALUATION_CELLS = 100_000
MAX_EVALUATION_WORK = 1_000_000
MAX_TOTAL_EVALUATION_WORK = 4_000_000


class EvidenceError(ValueError):
    """The normalized network evidence is malformed or too expensive to audit."""


def _integer(name: str, value: int, *, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise EvidenceError(f"{name} must be between {minimum} and {maximum}")


def _name(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{name} must be a non-empty string")
    if value != value.strip() or len(value) > 128:
        raise EvidenceError(f"{name} must be trimmed and at most 128 characters")
    if any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise EvidenceError(f"{name} must contain printable ASCII without spaces")


def _network(value: str) -> ipaddress.IPv4Network | ipaddress.IPv6Network:
    if not isinstance(value, str):
        raise EvidenceError("cidr must be a string")
    try:
        return ipaddress.ip_network(value, strict=True)
    except ValueError as exc:
        raise EvidenceError(f"invalid canonical CIDR: {value}") from exc


@dataclass(frozen=True, slots=True)
class AclRule:
    number: int
    action: str
    protocol: str
    cidr: str
    port_start: int
    port_end: int

    def __post_init__(self) -> None:
        _integer("rule number", self.number, minimum=1, maximum=32_766)
        if self.action not in {"allow", "deny"}:
            raise EvidenceError("rule action must be allow or deny")
        if self.protocol not in {"tcp", "udp", "all"}:
            raise EvidenceError("rule protocol must be tcp, udp, or all")
        _network(self.cidr)
        _integer("port_start", self.port_start, minimum=0, maximum=65_535)
        _integer("port_end", self.port_end, minimum=0, maximum=65_535)
        if self.port_end < self.port_start:
            raise EvidenceError("port_end must be at least port_start")
        if self.protocol == "all" and (self.port_start, self.port_end) != (0, 65_535):
            raise EvidenceError("all-protocol rules must cover ports 0 through 65535")

    @property
    def network(self) -> ipaddress.IPv4Network | ipaddress.IPv6Network:
        return _network(self.cidr)


@dataclass(frozen=True, slots=True)
class NetworkAcl:
    name: str
    ingress: tuple[AclRule, ...]
    egress: tuple[AclRule, ...]

    def __post_init__(self) -> None:
        _name("acl name", self.name)
        for direction, rules in (("ingress", self.ingress), ("egress", self.egress)):
            if not isinstance(rules, tuple):
                raise EvidenceError(f"{direction} must be a tuple of ACL rules")
            if len(rules) > MAX_RULES_PER_DIRECTION:
                raise EvidenceError(f"{direction} exceeds {MAX_RULES_PER_DIRECTION} rules")
            if not all(isinstance(rule, AclRule) for rule in rules):
                raise EvidenceError(f"{direction} must contain ACL rules")
            numbers = [rule.number for rule in rules]
            if len(numbers) != len(set(numbers)):
                raise EvidenceError(f"{direction} rule numbers must be unique")


@dataclass(frozen=True, slots=True)
class Segment:
    name: str
    cidr: str
    acl: str

    def __post_init__(self) -> None:
        _name("segment name", self.name)
        _network(self.cidr)
        _name("acl reference", self.acl)

    @property
    def network(self) -> ipaddress.IPv4Network | ipaddress.IPv6Network:
        return _network(self.cidr)


@dataclass(frozen=True, slots=True)
class RequiredFlow:
    name: str
    source_segment: str
    destination_segment: str
    protocol: str
    destination_port_start: int
    destination_port_end: int
    source_port_start: int
    source_port_end: int

    def __post_init__(self) -> None:
        _name("flow name", self.name)
        _name("source_segment", self.source_segment)
        _name("destination_segment", self.destination_segment)
        if self.protocol not in {"tcp", "udp"}:
            raise EvidenceError("flow protocol must be tcp or udp")
        for name, value in (
            ("destination_port_start", self.destination_port_start),
            ("destination_port_end", self.destination_port_end),
            ("source_port_start", self.source_port_start),
            ("source_port_end", self.source_port_end),
        ):
            _integer(name, value, minimum=0, maximum=65_535)
        if self.destination_port_end < self.destination_port_start:
            raise EvidenceError("destination port range is reversed")
        if self.source_port_end < self.source_port_start:
            raise EvidenceError("source port range is reversed")


@dataclass(frozen=True, slots=True)
class FlowViolation:
    flow_ref: str
    stage: str
    reason: str
    blocking_rule: int | None


@dataclass(frozen=True, slots=True)
class FlowResult:
    flow_ref: str
    passed: bool
    evaluated_cells: int


@dataclass(frozen=True, slots=True)
class NetworkAclReport:
    passed: bool
    segment_count: int
    acl_count: int
    flow_count: int
    evaluated_cells: int
    flows: tuple[FlowResult, ...]
    violations: tuple[FlowViolation, ...]
    evidence_digest: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = "stateless-network-acl-audit-v1"
        return payload


def _flow_ref(flow: RequiredFlow) -> str:
    material = f"{flow.name}\0{flow.source_segment}\0{flow.destination_segment}\0{flow.protocol}"
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def _intersection_bounds(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> tuple[int, int] | None:
    start = max(first_start, second_start)
    end = min(first_end, second_end)
    return (start, end) if start <= end else None


def _evaluate_direction(
    rules: tuple[AclRule, ...],
    *,
    peer: ipaddress.IPv4Network | ipaddress.IPv6Network,
    protocol: str,
    port_start: int,
    port_end: int,
) -> tuple[bool, str | None, int | None, int, int]:
    ordered = sorted(rules, key=lambda rule: rule.number)
    address_start = int(peer.network_address)
    address_end = int(peer.broadcast_address)
    address_boundaries = {address_start, address_end + 1}
    port_boundaries = {port_start, port_end + 1}

    relevant: list[AclRule] = []
    for rule in ordered:
        rule_network = rule.network
        if rule.protocol not in {protocol, "all"} or rule_network.version != peer.version:
            continue
        address_overlap = _intersection_bounds(
            address_start,
            address_end,
            int(rule_network.network_address),
            int(rule_network.broadcast_address),
        )
        port_overlap = _intersection_bounds(port_start, port_end, rule.port_start, rule.port_end)
        if address_overlap is None or port_overlap is None:
            continue
        relevant.append(rule)
        address_boundaries.update((address_overlap[0], address_overlap[1] + 1))
        port_boundaries.update((port_overlap[0], port_overlap[1] + 1))

    addresses = sorted(address_boundaries)
    ports = sorted(port_boundaries)
    cell_count = (len(addresses) - 1) * (len(ports) - 1)
    if cell_count > MAX_EVALUATION_CELLS:
        raise EvidenceError(f"ACL evaluation exceeds {MAX_EVALUATION_CELLS} address/port cells")
    work = cell_count * max(1, len(relevant))
    if work > MAX_EVALUATION_WORK:
        raise EvidenceError(f"ACL evaluation exceeds {MAX_EVALUATION_WORK} rule checks")

    for address_index in range(len(addresses) - 1):
        address_type = ipaddress.IPv4Address if peer.version == 4 else ipaddress.IPv6Address
        address = address_type(addresses[address_index])
        for port_index in range(len(ports) - 1):
            port = ports[port_index]
            decision: AclRule | None = None
            for rule in relevant:
                if address in rule.network and rule.port_start <= port <= rule.port_end:
                    decision = rule
                    break
            if decision is None:
                return False, "implicit_deny", None, cell_count, work
            if decision.action == "deny":
                return False, "explicit_deny", decision.number, cell_count, work
    return True, None, None, cell_count, work


def _canonical_payload(
    segments: Sequence[Segment],
    acls: Sequence[NetworkAcl],
    flows: Sequence[RequiredFlow],
) -> dict[str, Any]:
    return {
        "acls": [
            {
                "egress": [
                    asdict(rule) for rule in sorted(acl.egress, key=lambda item: item.number)
                ],
                "ingress": [
                    asdict(rule) for rule in sorted(acl.ingress, key=lambda item: item.number)
                ],
                "name": acl.name,
            }
            for acl in sorted(acls, key=lambda item: item.name)
        ],
        "flows": [asdict(flow) for flow in sorted(flows, key=lambda item: item.name)],
        "schema_version": "stateless-network-acl-evidence-v1",
        "segments": [asdict(segment) for segment in sorted(segments, key=lambda item: item.name)],
    }


def _digest(
    segments: Sequence[Segment],
    acls: Sequence[NetworkAcl],
    flows: Sequence[RequiredFlow],
) -> str:
    canonical = json.dumps(
        _canonical_payload(segments, acls, flows),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def audit_network_acls(
    segments: Sequence[Segment],
    acls: Sequence[NetworkAcl],
    flows: Sequence[RequiredFlow],
) -> NetworkAclReport:
    segment_values = tuple(segments)
    acl_values = tuple(acls)
    flow_values = tuple(flows)
    if not 1 <= len(segment_values) <= MAX_SEGMENTS:
        raise EvidenceError(f"segment count must be between 1 and {MAX_SEGMENTS}")
    if not 1 <= len(acl_values) <= MAX_ACLS:
        raise EvidenceError(f"ACL count must be between 1 and {MAX_ACLS}")
    if not 1 <= len(flow_values) <= MAX_FLOWS:
        raise EvidenceError(f"flow count must be between 1 and {MAX_FLOWS}")
    if not all(isinstance(value, Segment) for value in segment_values):
        raise EvidenceError("segments must contain Segment values")
    if not all(isinstance(value, NetworkAcl) for value in acl_values):
        raise EvidenceError("acls must contain NetworkAcl values")
    if not all(isinstance(value, RequiredFlow) for value in flow_values):
        raise EvidenceError("flows must contain RequiredFlow values")

    segments_by_name = {segment.name: segment for segment in segment_values}
    acls_by_name = {acl.name: acl for acl in acl_values}
    if len(segments_by_name) != len(segment_values):
        raise EvidenceError("segment names must be unique")
    if len(acls_by_name) != len(acl_values):
        raise EvidenceError("ACL names must be unique")
    if len({flow.name for flow in flow_values}) != len(flow_values):
        raise EvidenceError("flow names must be unique")
    for segment in segment_values:
        if segment.acl not in acls_by_name:
            raise EvidenceError(f"segment {segment.name} references an unknown ACL")
    for index, first in enumerate(segment_values):
        for second in segment_values[index + 1 :]:
            if first.network.version == second.network.version and first.network.overlaps(
                second.network
            ):
                raise EvidenceError("segment CIDRs must not overlap")

    violations: list[FlowViolation] = []
    results: list[FlowResult] = []
    total_cells = 0
    total_work = 0
    for flow in sorted(flow_values, key=lambda item: item.name):
        try:
            source = segments_by_name[flow.source_segment]
            destination = segments_by_name[flow.destination_segment]
        except KeyError as exc:
            raise EvidenceError("flow references an unknown segment") from exc
        if source.name == destination.name:
            raise EvidenceError("flow source and destination must differ")
        if source.network.version != destination.network.version:
            raise EvidenceError("flow segments must use the same IP family")
        source_acl = acls_by_name[source.acl]
        destination_acl = acls_by_name[destination.acl]
        reference = _flow_ref(flow)
        stages = (
            (
                "forward_source_egress",
                source_acl.egress,
                destination.network,
                flow.destination_port_start,
                flow.destination_port_end,
            ),
            (
                "forward_destination_ingress",
                destination_acl.ingress,
                source.network,
                flow.destination_port_start,
                flow.destination_port_end,
            ),
            (
                "return_destination_egress",
                destination_acl.egress,
                source.network,
                flow.source_port_start,
                flow.source_port_end,
            ),
            (
                "return_source_ingress",
                source_acl.ingress,
                destination.network,
                flow.source_port_start,
                flow.source_port_end,
            ),
        )
        passed = True
        flow_cells = 0
        for stage, rules, peer, port_start, port_end in stages:
            allowed, reason, blocking_rule, cells, work = _evaluate_direction(
                rules,
                peer=peer,
                protocol=flow.protocol,
                port_start=port_start,
                port_end=port_end,
            )
            flow_cells += cells
            total_work += work
            if total_work > MAX_TOTAL_EVALUATION_WORK:
                raise EvidenceError(f"audit exceeds {MAX_TOTAL_EVALUATION_WORK} total rule checks")
            if not allowed:
                passed = False
                violations.append(
                    FlowViolation(
                        flow_ref=reference,
                        stage=stage,
                        reason=reason or "implicit_deny",
                        blocking_rule=blocking_rule,
                    )
                )
        total_cells += flow_cells
        results.append(FlowResult(reference, passed, flow_cells))

    return NetworkAclReport(
        passed=not violations,
        segment_count=len(segment_values),
        acl_count=len(acl_values),
        flow_count=len(flow_values),
        evaluated_cells=total_cells,
        flows=tuple(results),
        violations=tuple(violations),
        evidence_digest=_digest(segment_values, acl_values, flow_values),
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise EvidenceError(f"{name} must contain exactly {', '.join(sorted(expected))}")


def _rule(value: Any) -> AclRule:
    if not isinstance(value, dict):
        raise EvidenceError("ACL rule must be an object")
    _exact_keys(
        value, {"action", "cidr", "number", "port_end", "port_start", "protocol"}, "ACL rule"
    )
    return AclRule(**value)


def _load(path: Path) -> tuple[list[Segment], list[NetworkAcl], list[RequiredFlow]]:
    if path.is_symlink():
        raise EvidenceError("input must not be a symlink")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise EvidenceError(f"input exceeds {MAX_INPUT_BYTES} bytes")
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            EvidenceError(f"non-finite JSON value: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise EvidenceError("input must be an object")
    _exact_keys(payload, {"acls", "required_flows", "segments"}, "input")
    if not all(isinstance(payload[key], list) for key in payload):
        raise EvidenceError("segments, acls, and required_flows must be arrays")

    segments: list[Segment] = []
    for value in payload["segments"]:
        if not isinstance(value, dict):
            raise EvidenceError("segment must be an object")
        _exact_keys(value, {"acl", "cidr", "name"}, "segment")
        segments.append(Segment(**value))

    acls: list[NetworkAcl] = []
    for value in payload["acls"]:
        if not isinstance(value, dict):
            raise EvidenceError("ACL must be an object")
        _exact_keys(value, {"egress", "ingress", "name"}, "ACL")
        if not isinstance(value["ingress"], list) or not isinstance(value["egress"], list):
            raise EvidenceError("ACL ingress and egress must be arrays")
        acls.append(
            NetworkAcl(
                name=value["name"],
                ingress=tuple(_rule(rule) for rule in value["ingress"]),
                egress=tuple(_rule(rule) for rule in value["egress"]),
            )
        )

    flows: list[RequiredFlow] = []
    flow_keys = {
        "destination_port_end",
        "destination_port_start",
        "destination_segment",
        "name",
        "protocol",
        "source_port_end",
        "source_port_start",
        "source_segment",
    }
    for value in payload["required_flows"]:
        if not isinstance(value, dict):
            raise EvidenceError("required flow must be an object")
        _exact_keys(value, flow_keys, "required flow")
        flows.append(RequiredFlow(**value))
    return segments, acls, flows


def _write_atomic(path: Path, content: str) -> None:
    if path.is_symlink():
        raise EvidenceError("output must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit stateless network ACLs against required bidirectional flows."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        report = audit_network_acls(*_load(arguments.input))
        rendered = json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n"
        if arguments.output:
            _write_atomic(arguments.output, rendered)
        else:
            print(rendered, end="")
    except (EvidenceError, json.JSONDecodeError, OSError, TypeError) as exc:
        print(f"network ACL audit failed: {exc}", file=sys.stderr)
        return INVALID_EVIDENCE_EXIT_CODE
    return PASS_EXIT_CODE if report.passed else POLICY_FAILURE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
