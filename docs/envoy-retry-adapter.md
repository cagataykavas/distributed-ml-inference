# Envoy retry-policy adapter

`envoy_retry_adapter.py` converts one explicitly selected static Envoy HTTP route and its upstream
cluster retry budget into the provider-neutral artifact consumed by `retry_topology_audit.py`. This
closes the gap between a reviewed policy document and the configuration that actually enables proxy
retries.

The implementation follows Envoy's v3 [route retry policy](https://www.envoyproxy.io/docs/envoy/latest/api-v3/config/route/v3/route_components.proto),
[router retry behavior](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/router_filter),
and [cluster retry-budget](https://www.envoyproxy.io/docs/envoy/latest/api-v3/config/cluster/v3/circuit_breaker.proto)
contracts. Envoy counts `num_retries` after the original attempt, applies the route timeout across
all attempts, and uses fully jittered exponential backoff. The adapter preserves those semantics in
the normalized attempt and timing evidence.

## Release use

Use JSON so duplicate fields can be rejected deterministically. Validate the full Envoy schema with
the same pinned Envoy binary that will run the configuration, then adapt and audit one route:

```bash
envoy --mode validate -c envoy.json

python envoy_retry_adapter.py envoy.json \
  --listener inference_listener \
  --virtual-host inference_api \
  --route inference.predict \
  --generated-at 2026-11-09T09:30:00Z \
  --max-amplification 2 \
  --max-retry-layers 1 \
  --max-retry-budget-percent 20 \
  --output artifacts/retry-policy.json

python retry_topology_audit.py artifacts/retry-policy.json \
  --output artifacts/retry-decision.json
```

The adapter exits `0` after an atomic policy write and `2` for malformed, ambiguous, unsupported,
or unsafe configuration. The downstream audit retains its separate pass (`0`), policy failure (`1`),
and untrusted-input (`2`) exits.

## Fail-closed subset

The selected listener must contain exactly one inline v3 HTTP connection manager. The selected
virtual host, named route, direct cluster, and cluster `DEFAULT` circuit-breaker threshold must each
be unique. Dynamic RDS, weighted/dynamic clusters, gRPC retry policies, host predicates, hedging,
and other unmodeled retry extensions are rejected.

The supported retry conditions are deliberately narrower than Envoy's full surface:

| Envoy configuration | Normalized condition |
|---|---|
| `connect-failure` | `connect_error`, `connect_timeout` |
| `reset`, `reset-before-request`, `refused-stream` | `reset` |
| `envoy-ratelimited` | `rate_limited` |
| `retriable-status-codes` with 429 | `rate_limited` |
| `retriable-status-codes` with 503 | `unavailable` |

Broad `5xx`, `gateway-error`, `retriable-4xx`, and other response policies fail closed. A 429 or 503
policy must explicitly configure `Retry-After` seconds handling through
`rate_limited_retry_back_off`. Exponential backoff must declare both base and maximum intervals.

The cluster must use `retry_budget` with an integer percentage, zero budget interval, and explicit
`min_retry_concurrency: 0`. This prevents Envoy's default minimum retry concurrency from exceeding
the declared percentage at low traffic. A simultaneous `max_retries` circuit breaker is rejected
instead of retained as dormant configuration.

## Runtime override and idempotency boundary

Envoy can accept downstream `x-envoy-*` headers that alter retry count, conditions, timeouts, status
codes, or hedging. The route, virtual host, or route configuration must remove every reviewed
override header before routing; otherwise adaptation fails.

A retried non-idempotent operation passes only when the route both requires an
`x-idempotency-key` header and carries this reviewed metadata:

```json
{
  "filter_metadata": {
    "retry_topology": {"idempotency_key_bound": true}
  }
}
```

The annotation remains an assertion: neither Envoy nor this adapter can prove that the application
atomically binds the key to the operation result. The downstream audit therefore fails POST/PATCH
retries when either half is absent.

## Evidence and limitations

The artifact's SHA-256 commits to the complete canonical Envoy JSON, and the loader rejects
duplicate fields, non-finite values, symlinks, non-regular files, read-time mutation, and inputs over
2 MiB. The adapter output contains route identifiers and normalized controls, not endpoints,
credentials, headers, or request data.

This is a static admission adapter, not live enforcement evidence. It does not query Envoy's admin
config dump or metrics, authenticate the configuration producer, prove that the selected artifact
was deployed, validate DNS/TLS/upstream identity, or measure retry-budget exhaustion. Production
follow-up should compare the accepted digest with the deployed config dump and correlate it with
low-cardinality retry-attempt, overflow, timeout, and budget-exhaustion metrics.
