"""Shared GRU encoder + per-horizon prediction heads (docs/server-docs/model-design.md section 1)."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class HorizonHead(nn.Module):
    """Small MLP head predicting (delta_peak, delta_dip) for one horizon."""

    def __init__(self, hidden_size: int, head_hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_size),
            nn.ReLU(),
            nn.Linear(head_hidden_size, 2),  # [delta_peak, delta_dip]
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.net(encoded)


class MultiHorizonGRUModel(nn.Module):
    """Shared GRU encoder feeding independent per-horizon heads.

    Input features are projected to `hidden_size` and run through a stacked GRU. The GRU's
    hidden state is carried across TBPTT chunks by the training loop (see model/train_loop.py),
    not by this module -- forward() always takes the incoming hidden state explicitly. Every
    per-step GRU output feeds every horizon head.
    """

    def __init__(
        self,
        n_features: int,
        horizons: list[int],
        hidden_size: int = 128,
        num_layers: int = 2,
        head_hidden_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizons = horizons
        self.input_proj = nn.Linear(n_features, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.heads = nn.ModuleDict(
            {str(w): HorizonHead(hidden_size, head_hidden_size) for w in horizons}
        )

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, chunk_len, n_features). Returns (preds, next_hidden).

        preds: (batch, chunk_len, n_horizons, 2) -- [delta_peak_W, delta_dip_W] per horizon,
        ordered as `self.horizons`.
        """
        projected = self.input_proj(x)
        encoded, next_hidden = self.gru(projected, hidden)
        preds = torch.stack([self.heads[str(w)](encoded) for w in self.horizons], dim=2)
        return preds, next_hidden

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        return torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size, device=device)
