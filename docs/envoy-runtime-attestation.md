# Envoy runtime retry-policy attestation

`envoy_runtime_attestation.py` closes one boundary left by the static retry-policy adapter: a
reviewed bootstrap file is not proof that a running proxy loaded the same retry controls. The tool
accepts a freshness-bound, HMAC-authenticated Envoy admin config dump, re-derives the selected route
and cluster retry policy, and compares its typed semantics with the reviewed policy artifact.

The result is intentionally narrow. A passing report proves that an authorized collector observed
an equivalent static retry policy within the configured freshness window. It does not prove traffic
used that Envoy process, that upstream requests succeeded, or that retry budgets behaved well.

## Collector envelope

The verifier first issues a unique 16–128 character URL-safe nonce. A collector fetches Envoy's
`/config_dump`, embeds it without transformation, and signs these exact canonical JSON fields:

```json
{
  "schema_version": 1,
  "captured_at": "2026-11-09T20:00:00Z",
  "nonce": "release-check-9f3a7c2d",
  "config_dump": {"configs": []}
}
```

The published envelope adds `hmac_sha256`. `sign_runtime_envelope` implements the canonicalization
and HMAC-SHA256 contract for a collector:

```python
import json
from pathlib import Path

from envoy_runtime_attestation import sign_runtime_envelope

fields = {
    "schema_version": 1,
    "captured_at": "2026-11-09T20:00:00Z",
    "nonce": "release-check-9f3a7c2d",
    "config_dump": json.loads(Path("config-dump.json").read_text()),
}
Path("runtime-envelope.json").write_text(
    json.dumps(sign_runtime_envelope(fields, Path("attestation.key").read_bytes()))
)
```

Generate a dedicated 32-byte key, provision it out of band to the collector and verifier, and keep
the verifier copy owner-only:

```bash
umask 077
openssl rand 32 > attestation.key
```

The key is not an application secret, TLS private key, or Envoy admin credential. Do not put it in
the repository, command line, artifact bundle, or report.

## Release gate

Run the static adapter and retry audit first, then challenge the collector and verify its envelope:

```bash
python envoy_runtime_attestation.py \
  artifacts/retry-policy.json \
  artifacts/runtime-envelope.json \
  --key-file /run/secrets/envoy-attestation.key \
  --expected-nonce release-check-9f3a7c2d \
  --listener inference_listener \
  --virtual-host inference_api \
  --route inference.predict \
  --max-age-seconds 60 \
  --output artifacts/runtime-attestation.json
```

Exit codes separate outcomes:

| Code | Meaning |
|---:|---|
| `0` | authenticated, fresh, and semantically equivalent |
| `1` | authenticated evidence shows reviewed retry-policy drift |
| `2` | malformed, stale, challenge-mismatched, unauthenticated, or unsupported evidence |

The report binds the reviewed config digest, runtime bootstrap digest, full config-dump digest,
operation, nonce, capture time, and drift codes. It contains no key material or full configuration.

## Fail-closed boundary

- Envelope and policy files must be singly linked regular files; symlinks and read-time mutation
  are rejected.
- The HMAC key must be 32–64 bytes and inaccessible to group/other users.
- The verifier requires the exact challenge nonce and rejects captures older than the configured
  window or more than five seconds in the future.
- JSON duplicate fields, non-finite numbers, unknown envelope fields, oversized inputs, and more
  than 128 config-dump sections are rejected.
- Exactly one `envoy.admin.v3.BootstrapConfigDump` is required. Its bootstrap is normalized through
  the same fail-closed adapter as the reviewed artifact.
- Operation semantics and every retry-layer control are compared as typed values. The reviewed
  audit limits are reused while normalizing runtime evidence; timestamps and whole-document digests
  are recorded but are not mistaken for semantic equality.

## Limitations and next step

HMAC authenticates possession of a shared collector key; it does not authenticate Envoy itself or
provide non-repudiation. A compromised collector can sign a fabricated dump. Nonce uniqueness must
be enforced by the external release system, because this one-shot verifier deliberately stores no
replay database. Only the static bootstrap dump is supported; dynamic xDS resources fail closed.

The next increment should compare this attestation with low-cardinality Envoy retry-attempt,
overflow, timeout, and budget-exhaustion counters over the same release window, while retaining the
distinction between configured policy and observed behavior.
