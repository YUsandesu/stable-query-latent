"""Variant of LatentQueryFlatRegressor that keeps only the FIRST learnable latent
array and derives every later query by linearly compressing the previous stage's
latents (instead of a fresh learnable query array per stage).

Pipeline (default query_sizes=(32, 16, 8)):

    input -> Linear(input_dim -> hidden_dim)
      stage 0:  learnable queries [32] --cross-attend--> projected input   => L0  (B,32,D)
                self-attention(L0)                                          => L0  (B,32,D)   [new]
      stage 1:  Linear reduce L0 over latent axis 32->16  => Q1
                Q1 --cross-attend--> L0 (first latent array)               => L1  (B,16,D)
      stage 2:  Linear reduce L1 over latent axis 16->8   => Q2
                Q2 --cross-attend--> L1 (previous cross-attention)         => L2  (B,8,D)
    head: flatten(8 x D) -> output

vs. the original, which uses an independent learnable query array at every stage.
Constructor signature matches LatentQueryFlatRegressor so it is a drop-in swap.
"""

import torch
from torch import nn


def _ffn(dim, mlp_ratio, dropout):
    hidden = int(dim * mlp_ratio)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, dim),
        nn.Dropout(dropout),
    )


class LatentArrayCrossAttention(nn.Module):
    """Cross-attention with a learnable query array (the 'latent array')."""

    def __init__(self, dim, num_queries, num_heads=8, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = _ffn(dim, mlp_ratio, dropout)

    def forward(self, context, key_padding_mask=None):
        batch_size = context.size(0)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attention(
            query=self.query_norm(queries),
            key=self.context_norm(context),
            value=self.context_norm(context),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        latents = queries + attended
        latents = latents + self.ffn(latents)
        return latents


class QueryCrossAttention(nn.Module):
    """Cross-attention whose query is provided externally (no learnable array)."""

    def __init__(self, dim, num_heads=8, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = _ffn(dim, mlp_ratio, dropout)

    def forward(self, query, context, key_padding_mask=None):
        attended, _ = self.attention(
            query=self.query_norm(query),
            key=self.context_norm(context),
            value=self.context_norm(context),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        latents = query + attended
        latents = latents + self.ffn(latents)
        return latents


class SelfAttention(nn.Module):
    """Standard pre-norm self-attention block over the latent set."""

    def __init__(self, dim, num_heads=8, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = _ffn(dim, mlp_ratio, dropout)

    def forward(self, x):
        normed = self.norm(x)
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        x = x + attended
        x = x + self.ffn(x)
        return x


class LatentReducer(nn.Module):
    """Reduce the number of latent vectors via a Linear over the latent axis:
    (B, num_in, D) -> (B, num_out, D). Each output query is a learned linear
    combination of the input latents."""

    def __init__(self, num_in, num_out):
        super().__init__()
        self.proj = nn.Linear(num_in, num_out)

    def forward(self, x):
        x = x.transpose(1, 2)   # (B, D, num_in)
        x = self.proj(x)        # (B, D, num_out)
        return x.transpose(1, 2)  # (B, num_out, D)


class FlatProjectionHead(nn.Module):
    def __init__(self, num_latents, hidden_dim, flat_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(num_latents * hidden_dim),
            nn.Linear(num_latents * hidden_dim, flat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, output_dim),
        )

    def forward(self, latents):
        return self.net(latents.flatten(start_dim=1))


class LatentQueryFunnelRegressor(nn.Module):
    """Single learnable latent array + self-attention, then linearly-reduced
    queries cascading through cross-attention. Drop-in for LatentQueryFlatRegressor."""

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

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Stage 0: the one and only learnable latent array, plus self-attention.
        self.cross0 = LatentArrayCrossAttention(hidden_dim, query_sizes[0], num_heads, dropout)
        self.self_attn = SelfAttention(hidden_dim, num_heads, dropout)

        # Stages 1..n: reduce previous latents -> queries, cross-attend to previous latents.
        self.reducers = nn.ModuleList(
            [LatentReducer(query_sizes[i - 1], query_sizes[i]) for i in range(1, len(query_sizes))]
        )
        self.cross_stages = nn.ModuleList(
            [QueryCrossAttention(hidden_dim, num_heads, dropout) for _ in range(1, len(query_sizes))]
        )

        self.head = FlatProjectionHead(
            num_latents=query_sizes[-1],
            hidden_dim=hidden_dim,
            flat_dim=flat_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(self, x, key_padding_mask=None):
        x = self.input_proj(x)

        latents = self.cross0(x, key_padding_mask=key_padding_mask)  # learnable Q attends to input
        latents = self.self_attn(latents)                            # refine the first latent array

        for reducer, cross in zip(self.reducers, self.cross_stages):
            query = reducer(latents)            # linearly compress current latents into queries
            latents = cross(query, latents)     # query the previous stage's latents
        return self.head(latents)


if __name__ == "__main__":
    model = LatentQueryFunnelRegressor(input_dim=1024, output_dim=50, hidden_dim=64, flat_dim=128)
    x = torch.randn(4, 20, 1024)
    print(model(x).shape)
    print("params:", sum(p.numel() for p in model.parameters()))
