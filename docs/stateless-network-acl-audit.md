# Stateless network ACL flow audit

Stateful security groups automatically admit return traffic, but subnet network ACLs do not. A
forward rule for ALB-to-inference traffic can therefore look correct while the reply path is
silently blocked, or an earlier deny can shadow a later allow for only part of a subnet.

`inference.network_acl` audits normalized subnet ACL evidence against required bidirectional flows.
For every flow it evaluates four independent stages:

1. source-subnet egress to the service port;
2. destination-subnet ingress from the client subnet;
3. destination-subnet egress to the client's declared source-port range; and
4. source-subnet ingress for that return traffic.

The evaluator preserves ordered first-match semantics. It partitions the complete required peer
CIDR and port range at every relevant rule boundary, then proves that each resulting cell reaches
an allow rule before any deny. Missing coverage is an implicit deny. This catches partial CIDR or
port coverage and shadowing without enumerating individual addresses.

Input is strict, bounded JSON containing `segments`, `acls`, and `required_flows`. Rule protocols
are `tcp`, `udp`, or `all`; all-protocol rules must cover ports 0–65535. Segment CIDRs must be
canonical and non-overlapping. Reports use hashed flow references and bind the complete normalized
topology to a deterministic SHA-256 digest.

```bash
python -m inference.network_acl topology.json --output network-acl-report.json
```

Exit code `0` means all required flows passed, `2` means valid evidence was denied, and `3` means
the evidence or operation was invalid. Inputs reject duplicate JSON keys, unknown fields,
non-finite values, symlinks, duplicate rule numbers, oversized files, and excessive evaluation
work. Output uses an fsynced temporary file and atomic replacement.

## Limitations

This is a static policy audit over normalized evidence. It does not authenticate a cloud export,
query live route tables, prove security-group behavior, inspect host firewalls, model NAT or
connection tracking, or establish that an application is listening. Protocol `all` intentionally
uses one port-independent rule model and ICMP type/code semantics are out of scope. The next step
is a signed AWS Network ACL adapter plus an isolated reachability canary that binds observed SYN,
SYN-ACK, and application-response evidence to the accepted topology digest.
