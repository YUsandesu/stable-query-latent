"""Recommendation-rate head for aggregated game-review embeddings."""

from __future__ import annotations

import torch
from torch import nn


class RecommendationRateHead(nn.Module):
    """Small MLP that predicts [positive_rate, negative_rate].

    The model returns raw logits by default. Use ``predict_rates`` to get a
    valid probability distribution whose two columns sum to 1.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (512, 128),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.LayerNorm(prev),
                    nn.Linear(prev, int(hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            prev = int(hidden_dim)
        layers.extend([nn.LayerNorm(prev), nn.Linear(prev, 2)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_rates(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)
