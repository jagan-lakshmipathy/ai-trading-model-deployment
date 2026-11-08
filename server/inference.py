"""Checkpoint loading + forward pass + denormalization (docs/server-docs/model-deployment-spec.md sections 2-4).

Wraps `MultiHorizonGRUModel` with the two serving-time concerns the raw model class doesn't
handle itself: picking the right checkpoint format (bare state_dict vs. the normalized variant's
`{"model_state", "feat_mean", "feat_std"}` dict) and converting raw model output (normalized
deltas) back into `(peak_price, dip_price)` per horizon.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import ModelConfig
from .encoder import MultiHorizonGRUModel

EPS = 1e-8  # must match the eps used when targets were normalized during training (model-design.md section 2)


@dataclass
class Prediction:
    horizon: int
    peak_price: float
    dip_price: float


class InferenceEngine:
    """Loads one checkpoint variant and runs single-step (or short-sequence) inference.

    Always runs on CPU by default -- batch=1 streaming inference gains nothing from a GPU (see
    docs/server-docs/plan.md discussion) and this keeps deployment simpler (no CUDA toolkit needed).
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_config: ModelConfig,
        n_features: int = 42,
        normalized: bool = False,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cpu")
        self.horizons = model_config.horizons
        self.normalized = normalized
        self.feat_mean: np.ndarray | None = None
        self.feat_std: np.ndarray | None = None

        self.model = MultiHorizonGRUModel(
            n_features=n_features,
            horizons=model_config.horizons,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_layers,
            head_hidden_size=model_config.head_hidden_size,
            dropout=model_config.dropout,
        ).to(self.device)

        if normalized:
            # weights_only=False: this checkpoint embeds numpy feat_mean/feat_std alongside the
            # state dict; safe here since it's produced by our own training pipeline, not an
            # untrusted source.
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state"])
            self.feat_mean = np.asarray(checkpoint["feat_mean"], dtype=np.float32)
            self.feat_std = np.asarray(checkpoint["feat_std"], dtype=np.float32)
        else:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.eval()  # dropout must be off for deterministic inference (spec section 1)

    def _prepare_features(self, features: np.ndarray) -> np.ndarray:
        if self.normalized:
            return (features - self.feat_mean) / self.feat_std
        return features

    @torch.no_grad()
    def forward(self, features: np.ndarray, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """features: (chunk_len, n_features) raw engineered feature rows for one ticker/session.

        Returns (preds, next_hidden). preds: (chunk_len, n_horizons, 2) normalized deltas,
        horizon order matches `self.horizons`.
        """
        prepared = self._prepare_features(features)
        x = torch.from_numpy(prepared.astype(np.float32)).unsqueeze(0).to(self.device)  # (1, chunk_len, n_features)
        preds, next_hidden = self.model(x, hidden)
        return preds.squeeze(0), next_hidden

    def denormalize(
        self, preds: torch.Tensor, close: float, realized_vol_15m: float
    ) -> list[Prediction]:
        """Invert the volatility-scaled delta transform (model-deployment-spec.md section 3).

        preds: (n_horizons, 2) normalized [delta_peak, delta_dip] per horizon, as returned by
        the last step of `forward` (typically `preds[-1]` for a multi-row call).
        """
        scale = close * realized_vol_15m + EPS
        preds_np = preds.detach().cpu().numpy()
        results = []
        for h_idx, horizon in enumerate(self.horizons):
            delta_peak, delta_dip = preds_np[h_idx]
            results.append(
                Prediction(
                    horizon=horizon,
                    peak_price=close + float(delta_peak) * scale,
                    dip_price=close + float(delta_dip) * scale,
                )
            )
        return results
