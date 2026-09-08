from __future__ import annotations

import secrets
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from batcher import BatcherClosed, DynamicBatcher, InferenceTimeout, QueueOverloaded
from inference.config import Settings
from inference.model import LinearRiskModel
from inference.runtime import ModelManager


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=4, max_length=4)


class PredictionResponse(BaseModel):
    score: float
    label: int
    model: str
    version: str
    latency_ms: float


class ReloadRequest(BaseModel):
    weights: tuple[float, float, float, float]
    bias: float
    name: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=100)


def create_app(
    settings: Settings | None = None,
    manager: ModelManager | None = None,
) -> FastAPI:
    config = settings or Settings.from_env()
    model_manager = manager or ModelManager(LinearRiskModel())
    registry = CollectorRegistry()
    requests = Counter(
        "inference_requests_total", "Total inference requests", ["status"], registry=registry
    )
    latency = Histogram(
        "inference_request_seconds",
        "End-to-end inference request latency",
        buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        registry=registry,
    )
    queue_depth = Gauge(
        "inference_queue_depth", "Requests waiting for batch admission", registry=registry
    )
    in_flight = Gauge(
        "inference_in_flight", "Requests currently executing in a model batch", registry=registry
    )
    model_generation = Gauge(
        "inference_model_generation", "Atomically loaded model generation", registry=registry
    )
    batcher: DynamicBatcher[list[float], dict[str, float | int | str]] = DynamicBatcher(
        model_manager.predict_batch,
        max_batch_size=config.max_batch_size,
        max_wait_ms=config.max_wait_ms,
        max_queue_size=config.max_queue_size,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await batcher.start()
        try:
            yield
        finally:
            await batcher.close(drain=True, timeout_seconds=config.shutdown_timeout_ms / 1000)

    application = FastAPI(
        title="Distributed ML Inference",
        version="0.3.0",
        description="Bounded, observable inference runtime with atomic model reloads.",
        lifespan=lifespan,
    )
    application.state.batcher = batcher
    application.state.model_manager = model_manager

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "alive"}

    @application.get("/ready")
    async def ready() -> dict[str, object]:
        if not batcher.is_ready:
            raise HTTPException(status_code=503, detail="batch runtime is not accepting work")
        return {
            "status": "ready",
            "model": model_manager.snapshot.metadata(),
            "runtime": batcher.snapshot(),
            "max_batch_size": config.max_batch_size,
            "max_wait_ms": config.max_wait_ms,
        }

    @application.get("/runtime")
    async def runtime() -> dict[str, object]:
        return {"model": model_manager.snapshot.metadata(), "batcher": batcher.snapshot()}

    @application.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest) -> PredictionResponse:
        started = time.perf_counter()
        try:
            result = await batcher.predict(
                request.features, timeout_seconds=config.request_timeout_ms / 1000
            )
            requests.labels(status="success").inc()
        except QueueOverloaded as exc:
            requests.labels(status="overloaded").inc()
            raise HTTPException(
                status_code=429,
                detail="inference capacity exhausted",
                headers={"Retry-After": "1"},
            ) from exc
        except InferenceTimeout as exc:
            requests.labels(status="timeout").inc()
            raise HTTPException(status_code=504, detail="inference deadline exceeded") from exc
        except BatcherClosed as exc:
            requests.labels(status="unavailable").inc()
            raise HTTPException(status_code=503, detail="inference runtime unavailable") from exc
        except ValueError as exc:
            requests.labels(status="invalid").inc()
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            requests.labels(status="error").inc()
            raise HTTPException(status_code=500, detail="inference failed") from exc

        elapsed = time.perf_counter() - started
        latency.observe(elapsed)
        return PredictionResponse(
            score=float(result["score"]),
            label=int(result["label"]),
            model=str(result["model"]),
            version=str(result["version"]),
            latency_ms=elapsed * 1000,
        )

    @application.post("/admin/models/reload")
    async def reload_model(
        request: ReloadRequest,
        x_admin_token: str | None = Header(default=None),
    ) -> dict[str, object]:
        if config.admin_token is None:
            raise HTTPException(status_code=404, detail="model reload is disabled")
        if x_admin_token is None or not secrets.compare_digest(x_admin_token, config.admin_token):
            raise HTTPException(status_code=403, detail="invalid admin token")
        candidate = LinearRiskModel(
            weights=request.weights,
            bias=request.bias,
            name=request.name,
            version=request.version,
        )
        try:
            snapshot = await model_manager.reload(candidate)
        except (ValueError, FloatingPointError) as exc:
            raise HTTPException(status_code=422, detail=f"candidate rejected: {exc}") from exc
        return {"status": "reloaded", "model": snapshot.metadata()}

    @application.get("/metrics")
    async def metrics() -> Response:
        runtime_snapshot = batcher.snapshot()
        queue_depth.set(int(runtime_snapshot["queue_depth"]))
        in_flight.set(int(runtime_snapshot["in_flight"]))
        model_generation.set(model_manager.snapshot.generation)
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return application


app = create_app()
