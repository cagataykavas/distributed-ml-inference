from __future__ import annotations

from pathlib import Path

import hcl2


def _resources() -> dict[tuple[str, str], dict]:
    with Path("infra/aws/main.tf").open(encoding="utf-8") as stream:
        document = hcl2.load(stream)
    resources: dict[tuple[str, str], dict] = {}
    for entry in document["resource"]:
        resource_type, instances = next(iter(entry.items()))
        resource_name, body = next(iter(instances.items()))
        resources[(resource_type, resource_name)] = body
    return resources


def test_private_egress_is_az_local_and_has_no_shared_nat_path() -> None:
    resources = _resources()

    public_subnets = resources[("aws_subnet", "public")]
    private_subnets = resources[("aws_subnet", "private")]
    nat_addresses = resources[("aws_eip", "nat")]
    nat = resources[("aws_nat_gateway", "nat")]
    private_routes = resources[("aws_route_table", "private")]
    private_associations = resources[("aws_route_table_association", "private")]

    assert public_subnets["count"] == "${local.availability_zone_count}"
    assert private_subnets["count"] == "${local.availability_zone_count}"
    assert nat_addresses["count"] == "${local.availability_zone_count}"
    assert nat["count"] == "${local.availability_zone_count}"
    assert nat["allocation_id"] == "${aws_eip.nat[count.index].id}"
    assert nat["subnet_id"] == "${aws_subnet.public[count.index].id}"
    assert private_routes["count"] == "${local.availability_zone_count}"
    assert private_routes["route"][0]["nat_gateway_id"] == (
        "${aws_nat_gateway.nat[count.index].id}"
    )
    assert private_associations["subnet_id"] == "${aws_subnet.private[count.index].id}"
    assert private_associations["count"] == "${local.availability_zone_count}"
    assert private_associations["route_table_id"] == ("${aws_route_table.private[count.index].id}")
