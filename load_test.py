from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@dataclass(frozen=True, slots=True)
class LoadReport:
    requested: int
    succeeded: int
    failed: int
    concurrency: int
    elapsed_seconds: float
    throughput_rps: float
    success_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    status_counts: dict[str, int]
    slo_passed: bool
    slo_failures: tuple[str, ...]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")


async def run_load(
    client: httpx.AsyncClient,
    *,
    url: str,
    requests: int,
    concurrency: int,
    max_p95_ms: float,
    min_success_rate: float,
) -> LoadReport:
    if requests <= 0 or concurrency <= 0:
        raise ValueError("requests and concurrency must be positive")
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    statuses: Counter[str] = Counter()

    async def guarded(index: int) -> None:
        async with semaphore:
            started = time.perf_counter()
            try:
                response = await client.post(
                    url,
                    json={"features": [index % 10 / 10.0, 0.2, 0.5, -0.1]},
                )
                statuses[str(response.status_code)] += 1
            except httpx.HTTPError:
                statuses["transport_error"] += 1
            finally:
                latencies.append((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    await asyncio.gather(*(guarded(index) for index in range(requests)))
    elapsed = time.perf_counter() - started
    succeeded = statuses["200"]
    success_rate = succeeded / requests
    p95 = percentile(latencies, 0.95)
    failures: list[str] = []
    if p95 > max_p95_ms:
        failures.append(f"p95 {p95:.2f}ms exceeds {max_p95_ms:.2f}ms")
    if success_rate < min_success_rate:
        failures.append(f"success rate {success_rate:.4f} is below {min_success_rate:.4f}")
    return LoadReport(
        requested=requests,
        succeeded=succeeded,
        failed=requests - succeeded,
        concurrency=concurrency,
        elapsed_seconds=elapsed,
        throughput_rps=requests / elapsed,
        success_rate=success_rate,
        latency_p50_ms=percentile(latencies, 0.50),
        latency_p95_ms=p95,
        latency_p99_ms=percentile(latencies, 0.99),
        latency_max_ms=max(latencies),
        status_counts=dict(sorted(statuses.items())),
        slo_passed=not failures,
        slo_failures=tuple(failures),
    )


async def run(args: argparse.Namespace) -> LoadReport:
    async with httpx.AsyncClient(timeout=args.timeout_seconds) as client:
        return await run_load(
            client,
            url=f"{args.base_url.rstrip('/')}/predict",
            requests=args.requests,
            concurrency=args.concurrency,
            max_p95_ms=args.max_p95_ms,
            min_success_rate=args.min_success_rate,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an SLO-gated inference load profile")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=float, default=10)
    parser.add_argument("--max-p95-ms", type=float, default=250)
    parser.add_argument("--min-success-rate", type=float, default=0.99)
    parser.add_argument("--output", type=Path, default=Path("artifacts/load-report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(run(args))
    report.write_json(args.output)
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    raise SystemExit(0 if report.slo_passed else 2)


if __name__ == "__main__":
    main()
