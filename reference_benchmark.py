from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from inference.config import Settings
from inference.service import create_app
from load_test import run_load


async def benchmark(output: Path, requests: int, concurrency: int) -> bool:
    app = create_app(
        Settings(max_batch_size=16, max_wait_ms=3, max_queue_size=128, request_timeout_ms=1_000)
    )
    async with app.router.lifespan_context(app):  # noqa: SIM117 - lifespan owns client scope.
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://benchmark"
        ) as client:
            report = await run_load(
                client,
                url="/predict",
                requests=requests,
                concurrency=concurrency,
                max_p95_ms=250,
                min_success_rate=1.0,
            )
    report.write_json(output)
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return report.slo_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic in-process CI benchmark")
    parser.add_argument("--output", type=Path, default=Path("artifacts/reference-load.json"))
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    passed = asyncio.run(benchmark(args.output, args.requests, args.concurrency))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
