# Retry topology admission

Retries can turn one failed inference request into a traffic spike. If a gateway, service client,
and transport each make three attempts, a single operation can reach the backend 27 times. This
audit makes that multiplication reviewable before rollout and emits a deterministic decision
artifact for CI or deployment policy.

The input orders retry controllers from the outermost caller to the innermost transport. It binds
the evidence to a SHA-256 digest of the reviewed configuration and evaluates:

- the product of attempts across every layer;
- a single-owner limit for retry responsibility;
- nested timeout windows against the end-to-end deadline;
- bounded exponential backoff with equal or full jitter;
- a per-layer retry budget cap;
- explicit transient retry conditions and `Retry-After` handling; and
- an idempotency key for retried `POST` and `PATCH` operations.

## Policy artifact

```json
{
  "schema_version": 1,
  "generated_at": "2026-10-01T12:00:00Z",
  "config_digest": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "operation": {
    "id": "inference.predict",
    "method": "POST",
    "deadline_ms": 2000,
    "idempotency_key_bound": true
  },
  "limits": {
    "max_amplification": 2,
    "max_retry_layers": 1,
    "max_retry_budget_percent": 20
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
      "honor_retry_after": true
    }
  ]
}
```

`retry_budget_percent` is the maximum share of normal traffic reserved for retries at that layer.
This audit checks the declared control; the runtime or proxy must still enforce it with a token
bucket or equivalent mechanism.

```bash
python retry_topology_audit.py retry-policy.json \
  --output artifacts/retry-decision.json
```

Exit `0` means the policy passed. Exit `1` means the JSON was valid but at least one invariant
failed. Exit `2` means the evidence could not be trusted: malformed JSON, duplicate or unknown
keys, unsupported retry conditions, invalid digests, and oversized inputs fail closed.

The decision records effective attempts, retry-layer count, worst-case retry window, and sorted
violations. No request is sent and no availability claim is inferred from the policy alone.

## Timing model and limitation

For a layer with `n` attempts, the audit adds `n` bounded invocation windows plus the maximum
exponential backoff between attempts. Working from the transport outward, each parent timeout must
contain the complete inner retry window. This is intentionally conservative and assumes sequential
attempts; hedging and parallel fan-out need a separate concurrency-aware model.

The next integration step is to export the same policy from the HTTP client or service-mesh config,
then compare declared retry budgets with live retry-attempt and budget-exhaustion metrics.
