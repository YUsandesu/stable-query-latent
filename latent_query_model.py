import torch
from torch import nn


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_queries,
        num_heads=8,
        dropout=0.1,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, context):
        batch_size = context.size(0)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        attended, _ = self.attention(
            query=self.query_norm(queries),
            key=self.context_norm(context),
            value=self.context_norm(context),
            need_weights=False,
        )

        latents = queries + attended
        latents = latents + self.ffn(latents)
        return latents


class FlatProjectionHead(nn.Module):
    def __init__(
        self,
        num_latents,
        hidden_dim,
        flat_dim,
        output_dim,
        dropout=0.1,
    ):
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
        flat = latents.flatten(start_dim=1)
        return self.net(flat)


class LatentQueryFlatRegressor(nn.Module):
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
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=hidden_dim,
                    num_queries=num_queries,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for num_queries in query_sizes
            ]
        )
        self.head = FlatProjectionHead(
            num_latents=query_sizes[-1],
            hidden_dim=hidden_dim,
            flat_dim=flat_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.input_proj(x)

        for index, block in enumerate(self.blocks):
            x = block(x)

        return self.head(x)


if __name__ == "__main__":
    batch_size = 4
    sequence_length = 20
    input_dim = 1024
    output_dim = 8

    model = LatentQueryFlatRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=256,
        flat_dim=512,
    )
    inputs = torch.randn(batch_size, sequence_length, input_dim)
    outputs = model(inputs)
    print(outputs.shape)
