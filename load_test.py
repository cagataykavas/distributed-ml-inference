from __future__ import annotations

import argparse
import asyncio
import time
from statistics import mean, quantiles

import httpx


async def run_request(client: httpx.AsyncClient, url: str, index: int) -> float:
    started = time.perf_counter()
    response = await client.post(
        url,
        json={"features": [index % 10 / 10.0, 0.2, 0.5, -0.1]},
    )
    response.raise_for_status()
    return (time.perf_counter() - started) * 1000


async def run(base_url: str, requests: int, concurrency: int) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []

    async with httpx.AsyncClient(timeout=10) as client:

        async def guarded(index: int) -> None:
            async with semaphore:
                latencies.append(await run_request(client, f"{base_url}/predict", index))

        started = time.perf_counter()
        await asyncio.gather(*(guarded(i) for i in range(requests)))
        elapsed = time.perf_counter() - started

    sorted_latencies = sorted(latencies)
    p95 = (
        quantiles(sorted_latencies, n=100, method="inclusive")[94]
        if len(latencies) > 1
        else latencies[0]
    )
    print(f"requests:       {requests}")
    print(f"concurrency:    {concurrency}")
    print(f"throughput_rps: {requests / elapsed:.2f}")
    print(f"latency_mean:   {mean(latencies):.2f} ms")
    print(f"latency_p95:    {p95:.2f} ms")
    print(f"latency_max:    {max(latencies):.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(run(args.base_url.rstrip("/"), args.requests, args.concurrency))


if __name__ == "__main__":
    main()
