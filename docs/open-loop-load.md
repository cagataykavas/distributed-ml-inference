# Open-loop inference load evidence

`python -m inference.open_loop` sends requests according to a fixed arrival schedule instead of
waiting for each previous request to complete. This matters for networked inference services because
a closed-loop client naturally slows down when the service slows down. That feedback can hide queue
growth and produce reassuring latency percentiles precisely when the service is overloaded—a
measurement error commonly called coordinated omission.

The repository's development extra supplies the HTTP load-driver dependency:

```bash
pip install -e '.[dev]'
```

The existing `load_test.py` remains useful for a bounded-concurrency client view and deterministic CI
regression. The open-loop profile answers a different question: **can the endpoint sustain a declared
arrival rate without client-side backlog, scheduler drift, or dropped arrivals?**

## Run a profile

```bash
python -m inference.open_loop \
  --base-url http://localhost:8000 \
  --requests 2000 \
  --rate-rps 250 \
  --max-in-flight 100 \
  --max-scheduled-p95-ms 250 \
  --max-scheduler-lag-p95-ms 20 \
  --min-success-rate 0.99 \
  --max-client-drop-rate 0.01 \
  --output artifacts/open-loop-load-report.json
```

Proxy-related environment variables are ignored by default so an ambient developer or CI proxy does
not silently change the measured path. Pass `--trust-env` only when the intended test topology
explicitly includes `HTTP_PROXY`, `HTTPS_PROXY`, certificate, or other HTTP-client environment
configuration.

Exit code `0` means every configured SLO passed. Exit code `2` means the profile completed but at
least one latency, scheduler, success-rate, or client-drop policy failed. Exit code `3` means the
command line or profile configuration was unsafe or malformed. Transport exception text is never
copied into the report.

The target is always `<base-url>/predict`. URL credentials, query parameters, and fragments are
rejected. The runner is intended for an operator-approved endpoint; it is not an SSRF-safe public
probing service and does not perform DNS or destination admission.

## Interpreting the report

- `service_latency_*` measures from actual HTTP dispatch until a response or transport failure.
- `scheduler_lag_*` measures how late the load generator dispatched relative to the planned arrival.
- `scheduled_latency_*` measures from the planned arrival until completion. This is the primary
  latency gate because it includes load-generator delay that a service-only timer would hide.
- `client_dropped` counts arrivals discarded because `max_in_flight` was already occupied. These
  requests are included in the success-rate denominator and have their own drop-rate SLO.
- `completion_throughput_rps` is successful completions divided by total wall-clock duration. It is
  not presented as model capacity; the declared `target_rate_rps` is the offered schedule.
- `evidence_sha256` content-addresses the complete report except for the digest itself.

The runner keeps at most `max_in_flight` request tasks alive. When that bound is full, it records a
client-side overload drop rather than creating a hidden waiter queue. The HTTP connection-pool limit
is set to the same value so a larger implicit pool cannot invalidate the task budget. Request count,
arrival rate, concurrency, timeout, and SLO values are finite and bounded before network traffic
begins. Evidence is written through a same-directory temporary file and atomically renamed so
interrupted runs do not leave a plausible partial report.

## Calibration and trust boundaries

- Run the generator from a controlled host with spare CPU, file-descriptor, ephemeral-port, and
  network capacity. A failing scheduler-lag SLO means the driver cannot support the requested profile;
  it does not prove the service is slow.
- Start below expected capacity, hold each rate long enough to reach steady state, then increase in
  explicit steps. Warm-up and measurement windows should be separate in formal benchmarks.
- Preserve non-200 status codes. A fast 429/503 response is still a failed inference request.
- Compare server-side queue, batch, saturation, and model metrics using synchronized clocks. This
  report alone cannot localize delay to DNS, TCP, TLS, proxy, queue, or model execution.
- The default HTTP client may reuse connections. This profile measures request handling over a warm
  connection pool, not new-connection or TLS-handshake capacity.
- A single load generator is not evidence of multi-region availability, production instance capacity,
  or statistically stable tail latency. Repeat runs and retain the environment and deployment identity
  alongside the JSON artifact.

The next increment is a staged profile runner with explicit warm-up/steady-state phases and trace-ID
correlation to server-side queue and batch spans.
