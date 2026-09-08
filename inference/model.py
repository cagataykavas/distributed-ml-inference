from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ModelAdapter(Protocol):
    name: str
    version: str

    def predict_batch(self, rows: list[list[float]]) -> list[dict[str, float | int | str]]: ...


@dataclass
class LinearRiskModel:
    """Deterministic local model used to exercise the serving architecture.

    The service boundary is intentionally model-agnostic. An ONNX Runtime,
    PyTorch or TensorRT adapter can implement the same ``predict_batch`` method.
    """

    weights: tuple[float, ...] = (0.65, -0.25, 0.45, 0.15)
    bias: float = -0.20
    name: str = "synthetic-risk-model"
    version: str = "2026.08"

    def predict_batch(self, rows: list[list[float]]) -> list[dict[str, float | int | str]]:
        if not rows:
            return []
        matrix = np.asarray(rows, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.weights):
            raise ValueError(f"expected {len(self.weights)} features per request")
        if not np.isfinite(matrix).all():
            raise ValueError("features must contain only finite numbers")
        parameters = np.asarray((*self.weights, self.bias), dtype=float)
        if not np.isfinite(parameters).all():
            raise ValueError("model parameters must contain only finite numbers")
        logits = matrix @ np.asarray(self.weights, dtype=float) + self.bias
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return [
            {
                "score": float(probability),
                "label": int(probability >= 0.5),
                "model": self.name,
                "version": self.version,
            }
            for probability in probabilities
        ]
