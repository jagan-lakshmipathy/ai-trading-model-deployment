"""Model architecture configuration (docs/server-docs/model-deployment-spec.md section 4)."""
from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass(frozen=True)
class ModelConfig:
    horizons: list[int] = field(default_factory=lambda: [5, 10, 15])
    hidden_size: int = 128
    num_layers: int = 2
    head_hidden_size: int = 64
    dropout: float = 0.1


def load_model_config(path: str) -> ModelConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return ModelConfig(**raw.get("model", {}))

