"""MLP-head baseline for the PXI latent-query benchmark.

This keeps only the first learnable latent array plus self-attention, then
funnels each latent's feature dimension through a shared MLP before the final
projection. It preserves the trainer-facing constructor signature used by the
other latent-query regressors.
"""

import sys
from pathlib import Path

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from latent_query_model_v2 import LatentArrayCrossAttention, SelfAttention


class FeatureFunnelHead(nn.Module):
    """Reduce each latent's feature dim through a shared MLP, then project."""

    def __init__(self, num_latents, hidden_dim, feature_dims, output_dim, dropout=0.1):
        super().__init__()
        layers = []
        prev = hidden_dim
        for width in feature_dims:
            layers += [nn.LayerNorm(prev), nn.Linear(prev, width), nn.GELU(), nn.Dropout(dropout)]
            prev = width
        self.feature_mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.out = nn.Sequential(
            nn.LayerNorm(num_latents * prev),
            nn.Linear(num_latents * prev, output_dim),
        )

    def forward(self, latents):
        latents = self.feature_mlp(latents)
        return self.out(latents.flatten(start_dim=1))


class LatentQueryBaseRegressor(nn.Module):
    """Base ablation: first latent array + self-attention + MLP feature funnel.

    ``query_sizes[0]`` controls the number of learnable latents. Remaining
    entries, such as ``(16, 8)`` in the default ``(32, 16, 8)``, become shared
    feature-MLP widths for every latent.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=256,
        flat_dim=512,
        query_sizes=(32, 16, 8),
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        if len(query_sizes) < 1:
            raise ValueError("query_sizes must have at least one entry.")

        num_latents = query_sizes[0]
        feature_dims = list(query_sizes[1:])

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cross0 = LatentArrayCrossAttention(hidden_dim, num_latents, num_heads, dropout)
        self.self_attn = SelfAttention(hidden_dim, num_heads, dropout)
        self.head = FeatureFunnelHead(num_latents, hidden_dim, feature_dims, output_dim, dropout)

    def forward(self, x, key_padding_mask=None):
        x = self.input_proj(x)
        latents = self.cross0(x, key_padding_mask=key_padding_mask)
        latents = self.self_attn(latents)
        return self.head(latents)


if __name__ == "__main__":
    model = LatentQueryBaseRegressor(input_dim=1024, output_dim=50, hidden_dim=64, flat_dim=128)
    x = torch.randn(4, 20, 1024)
    print("base out:", tuple(model(x).shape))
    print("params:", sum(p.numel() for p in model.parameters()))
