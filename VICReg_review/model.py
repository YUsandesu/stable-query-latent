"""Latent-array VICReg model with a GRL sentiment adversary.

The encoder returns 16 latent vectors by default, each with dimension 1024, so
the frozen SST MLP4-A head can be applied to every latent vector directly.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class _GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def gradient_reverse(x, lambda_=1.0):
    return _GradientReverseFn.apply(x, float(lambda_))


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x):
        return gradient_reverse(x, self.lambda_)


def _make_mlp(dim, hidden_dim, dropout):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )


class LatentSelfAttentionMLPBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = _make_mlp(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        normed = self.norm(x)
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        x = x + attended
        x = x + self.mlp(x)
        return x


class LatentArrayMLP(nn.Module):
    """Shared-weight view encoder for review subsets.

    Input:
        x: (batch, sentence_count, input_dim)
        key_padding_mask: optional bool mask, True where x is padding

    Output:
        (batch, num_latents, latent_dim), default (batch, 16, 1024)
    """

    def __init__(
        self,
        input_dim=1024,
        latent_dim=1024,
        num_latents=16,
        num_heads=8,
        depth=2,
        mlp_ratio=2.0,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.num_latents = int(num_latents)

        self.input_norm = nn.LayerNorm(input_dim)
        if input_dim == latent_dim:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Linear(input_dim, latent_dim)

        self.latent_array = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        self.query_norm = nn.LayerNorm(latent_dim)
        self.context_norm = nn.LayerNorm(latent_dim)
        self.cross_attention = nn.MultiheadAttention(
            latent_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_mlp = _make_mlp(latent_dim, int(latent_dim * mlp_ratio), dropout)
        self.blocks = nn.ModuleList(
            [
                LatentSelfAttentionMLPBlock(
                    latent_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, x, key_padding_mask=None):
        x = self.input_proj(self.input_norm(x))
        batch_size = x.size(0)
        queries = self.latent_array.unsqueeze(0).expand(batch_size, -1, -1)

        context = self.context_norm(x)
        attended, _ = self.cross_attention(
            query=self.query_norm(queries),
            key=context,
            value=context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        latents = queries + attended
        latents = latents + self.cross_mlp(latents)
        for block in self.blocks:
            latents = block(latents)
        return self.output_norm(latents)


Latent_Array_MLP = LatentArrayMLP


class Mlp4SentimentHead(nn.Module):
    """SST MLP4-A: 1024 -> 128 -> 32 -> 8 -> 1 with sigmoid output."""

    def __init__(self, input_dim=1024, hidden_dims=(128, 32, 8), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(prev, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden_dim
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_mlp4_a_sentiment_head(checkpoint_path, map_location="cpu", freeze=True):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = Mlp4SentimentHead()
    model.load_state_dict(state_dict)
    model.eval()
    if freeze:
        for param in model.parameters():
            param.requires_grad_(False)
    return model


def _off_diagonal(x):
    rows, cols = x.shape
    if rows != cols:
        raise ValueError("_off_diagonal expects a square matrix.")
    return x.flatten()[:-1].view(rows - 1, cols + 1)[:, 1:].flatten()


def vicreg_loss(
    z_a,
    z_b,
    invariance_weight=25.0,
    variance_weight=25.0,
    covariance_weight=1.0,
    eps=1e-4,
):
    """VICReg loss for two latent-array views.

    Invariance is computed on matching latent positions. Variance and covariance
    treat all latent vectors in the batch as the sample axis, which keeps the
    covariance matrix at 1024 x 1024 instead of flattening 16 x 1024 features.
    """

    if z_a.shape != z_b.shape:
        raise ValueError(f"VICReg views must have matching shapes, got {z_a.shape} and {z_b.shape}.")

    z_a = z_a.float()
    z_b = z_b.float()
    repr_loss = F.mse_loss(z_a, z_b)
    flat_a = z_a.reshape(-1, z_a.size(-1))
    flat_b = z_b.reshape(-1, z_b.size(-1))

    def variance_term(z):
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        return torch.mean(F.relu(1.0 - std))

    def covariance_term(z):
        sample_count = z.size(0)
        if sample_count < 2:
            return z.new_tensor(0.0)
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (sample_count - 1)
        return _off_diagonal(cov).pow(2).sum() / z.size(1)

    std_loss = 0.5 * (variance_term(flat_a) + variance_term(flat_b))
    cov_loss = 0.5 * (covariance_term(flat_a) + covariance_term(flat_b))
    total = (
        invariance_weight * repr_loss
        + variance_weight * std_loss
        + covariance_weight * cov_loss
    )
    return {
        "loss": total,
        "invariance": repr_loss,
        "variance": std_loss,
        "covariance": cov_loss,
    }


class SentimentAdversarialLoss(nn.Module):
    """GRL loss that pushes latent vectors toward SST-head uncertainty.

    The SST head is a frozen 0..1 regressor. We use Bernoulli entropy as a
    confidence surrogate. Minimizing entropy after a GRL makes the encoder ascend
    that entropy, so the frozen sentiment head is driven toward uncertainty.
    """

    def __init__(self, sentiment_head, grl_lambda=1.0, eps=1e-6, normalize=True):
        super().__init__()
        self.sentiment_head = sentiment_head
        self.grl = GradientReversal(grl_lambda)
        self.eps = eps
        self.normalize = normalize

    def forward(self, latents):
        flat = latents.reshape(-1, latents.size(-1))
        if self.normalize:
            flat = F.normalize(flat, p=2, dim=-1)
        pred = self.sentiment_head(self.grl(flat)).clamp(self.eps, 1.0 - self.eps)
        pred = pred.float()
        entropy = -(pred * pred.log() + (1.0 - pred) * (1.0 - pred).log())
        loss = entropy.mean()
        with torch.no_grad():
            stats = {
                "sentiment_mean": pred.mean(),
                "sentiment_std": pred.std(unbiased=False),
                "sentiment_entropy": entropy.mean(),
            }
        return loss, stats
