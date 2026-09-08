from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from batcher import DynamicBatcher
from inference.model import LinearRiskModel


REQUESTS = Counter("inference_requests_total", "Total inference requests", ["status"])
LATENCY = Histogram(
    "inference_request_seconds",
    "End-to-end inference request latency",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
QUEUE_DEPTH = Gauge("inference_queue_depth", "Current dynamic-batcher queue depth")
BATCH_SIZE = Histogram(
    "inference_batch_size",
    "Number of requests executed in one model batch",
    buckets=(1, 2, 4, 8, 16, 32, 64),
)


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=4, max_length=4)


class PredictionResponse(BaseModel):
    score: float
    label: int
    model: str
    version: str
    latency_ms: float


model = LinearRiskModel()


def predict_batch(rows: list[list[float]]) -> list[dict[str, float | int | str]]:
    BATCH_SIZE.observe(len(rows))
    return model.predict_batch(rows)


batcher: DynamicBatcher[list[float], dict[str, float | int | str]] = DynamicBatcher(
    predict_batch,
    max_batch_size=32,
    max_wait_ms=5,
)
worker_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global worker_task
    worker_task = asyncio.create_task(batcher.serve(), name="dynamic-inference-batcher")
    try:
        yield
    finally:
        if worker_task is not None:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass


app = FastAPI(
    title="Distributed ML Inference",
    version="0.2.0",
    description=(
        "Async inference API with dynamic micro-batching, model-version metadata, "
        "Prometheus metrics and cloud deployment examples."
    ),
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, object]:
    alive = worker_task is not None and not worker_task.done()
    if not alive:
        raise HTTPException(status_code=503, detail="batch worker is not running")
    return {
        "status": "ready",
        "model": model.name,
        "version": model.version,
        "max_batch_size": batcher.max_batch_size,
        "max_wait_ms": int(batcher.max_wait * 1000),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    started = time.perf_counter()
    QUEUE_DEPTH.inc()
    try:
        result = await batcher.predict(request.features)
        REQUESTS.labels(status="success").inc()
    except ValueError as exc:
        REQUESTS.labels(status="invalid").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=500, detail="inference failed") from exc
    finally:
        QUEUE_DEPTH.dec()

    elapsed = time.perf_counter() - started
    LATENCY.observe(elapsed)
    return PredictionResponse(
        score=float(result["score"]),
        label=int(result["label"]),
        model=str(result["model"]),
        version=str(result["version"]),
        latency_ms=elapsed * 1000,
    )


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
