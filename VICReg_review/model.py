"""Latent-array VICReg model with a GRL sentiment adversary.

The encoder projects the 1024-d input down to a wide latent_dim (256 by default),
runs one latent-query cross-attention layer at that width, then funnels each
latent vector down to a compact output_dim. There is no latent-to-latent
self-attention block in the current VICReg encoder. The H5 trainer pools the
latent slots into a game-level centroid before the VICReg variance/covariance
terms, then projects that centroid through an expander MLP so inter-game
separation is regularized at high width while downstream code can keep using the
compact centroid.

Because the frozen SST MLP4-A head still expects 1024-d inputs, the adversary
holds a learnable up-projection probe (output_dim -> 1024) placed AFTER the GRL,
so the encoder is always the adversarial party.
"""

from pathlib import Path
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


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


def _make_funnel_mlp(dims, dropout):
    """Sequential Linear funnel through dims, e.g. [256, 128, 64, 32, 18].

    GELU + Dropout between hidden layers; the final projection is raw (no
    activation, no norm) so VICReg's variance term controls the output scale.
    """
    layers = []
    for index in range(len(dims) - 1):
        layers.append(nn.Linear(dims[index], dims[index + 1]))
        if index < len(dims) - 2:
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class LatentArrayMLP(nn.Module):
    """Minimal shared-weight view encoder for review subsets.

    One cross-attention layer (learnable query array attends to the sentences),
    no residuals and no extra blocks, then a per-latent funnel down to output_dim.

    Input:
        x: (batch, sentence_count, input_dim)
        key_padding_mask: optional bool mask, True where x is padding

    Output:
        (batch, num_latents, output_dim), default (batch, 256, 18)
    """

    def __init__(
        self,
        input_dim=1024,
        latent_dim=256,
        num_latents=256,
        num_heads=8,
        dropout=0.1,
        output_dim=18,
        reduce_hidden=(128, 64, 32),
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.num_latents = int(num_latents)
        self.output_dim = int(output_dim)

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
        self.output_norm = nn.LayerNorm(latent_dim)
        self.reduce = _make_funnel_mlp([latent_dim, *reduce_hidden, self.output_dim], dropout)

    def forward_stem(self, x, key_padding_mask=None):
        context = self.context_norm(self.input_proj(self.input_norm(x)))
        queries = self.latent_array.unsqueeze(0).expand(x.size(0), -1, -1)

        latents, _ = self.cross_attention(
            query=self.query_norm(queries),
            key=context,
            value=context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        latents = self.output_norm(latents)
        return latents

    def forward_tail(self, latents):
        return self.reduce(latents)

    def forward(self, x, key_padding_mask=None):
        return self.forward_tail(self.forward_stem(x, key_padding_mask=key_padding_mask))


Latent_Array_MLP = LatentArrayMLP


def _heads_for_dim(dim, requested):
    requested = max(1, int(requested))
    dim = int(dim)
    if dim % requested == 0:
        return requested
    for heads in range(min(requested, dim), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


class LatentSelfAttentionBlock(nn.Module):
    """Pre-norm self-attention + FFN block over latent slots."""

    def __init__(self, dim, num_heads=8, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim,
            _heads_for_dim(dim, num_heads),
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        hidden = max(dim, int(round(dim * float(mlp_ratio))))
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        y = self.norm1(x)
        y, _ = self.attention(y, y, y, need_weights=False)
        x = x + self.dropout(y)
        x = x + self.ffn(self.norm2(x))
        return x


class LatentReductionCrossAttention(nn.Module):
    """Reduce latent width, then query the previous latent array at that width."""

    def __init__(self, input_dim, output_dim, num_heads=8, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.query_proj = nn.Linear(self.input_dim, self.output_dim)
        self.context_proj = nn.Linear(self.input_dim, self.output_dim)
        self.query_norm = nn.LayerNorm(self.output_dim)
        self.context_norm = nn.LayerNorm(self.output_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.output_dim,
            _heads_for_dim(self.output_dim, num_heads),
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.self_attention = LatentSelfAttentionBlock(
            self.output_dim,
            num_heads=num_heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
        )
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(self, latents):
        query = self.query_proj(latents)
        context = self.context_proj(latents)
        attended, _ = self.cross_attention(
            query=self.query_norm(query),
            key=self.context_norm(context),
            value=self.context_norm(context),
            need_weights=False,
        )
        latents = query + self.dropout(attended)
        latents = self.self_attention(latents)
        return self.output_norm(latents)


class HierarchicalLatentArrayMLP(nn.Module):
    """Latent-array encoder with self-attention and attentional reductions.

    Stage 0 learns a latent array that cross-attends to input sentences and then
    refines the latent slots with self-attention. Each later stage linearly
    reduces the latent width and uses the reduced slots as queries over the
    previous latent array, followed by another self-attention block. The output
    shape matches LatentArrayMLP: (batch, num_latents, output_dim).
    """

    def __init__(
        self,
        input_dim=1024,
        latent_dim=256,
        num_latents=256,
        num_heads=8,
        dropout=0.1,
        output_dim=64,
        reduce_hidden=(128,),
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.num_latents = int(num_latents)
        self.output_dim = int(output_dim)
        self.reduce_hidden = tuple(int(dim) for dim in reduce_hidden)

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
            _heads_for_dim(latent_dim, num_heads),
            dropout=0.0,  # no attention-weight dropout: it would break exact sentence-chunking
            batch_first=True,
        )
        # Regularization moved off the (chunked) sentence axis onto the latent
        # array: element-wise dropout of the learnable queries. This keeps the
        # stem exactly chunkable (the mask is drawn once, outside the chunk loop)
        # while still preventing latent-slot co-adaptation.
        self.latent_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.self_attention = LatentSelfAttentionBlock(
            latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
        )
        self.output_norm = nn.LayerNorm(latent_dim)

        dims = [latent_dim, *self.reduce_hidden, self.output_dim]
        self.reduction_stages = nn.ModuleList(
            [
                LatentReductionCrossAttention(
                    dims[index],
                    dims[index + 1],
                    num_heads=num_heads,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                )
                for index in range(len(dims) - 1)
            ]
        )
        # Default sentence-chunk size for the stem cross-attention. None = no
        # chunking (the trainer sets this from the per-combo memory plan so
        # forward_view/split_recompute pick it up without extra plumbing).
        self._stem_chunk_size = None

    def _cross_attend_chunked(self, x, queries, chunk_size):
        """Sentence->latent cross-attention computed in key-chunks.

        Same math as ``self.cross_attention`` (it reuses its in/out projection
        weights), but the per-sentence context pipeline + attention scores are
        recomputed per chunk under ``checkpoint`` during training, so peak
        activation memory is O(chunk + num_latents) instead of O(num_sentences).
        Uses the FlashAttention online-softmax merge, so the result is exact
        (up to floating-point reordering) regardless of ``chunk_size``.
        """
        mha = self.cross_attention
        bsz, n_sentences, _ = x.shape
        dim = self.latent_dim
        heads = mha.num_heads
        head_dim = dim // heads
        scale = 1.0 / math.sqrt(head_dim)
        w_q, w_k, w_v = mha.in_proj_weight.split(dim, dim=0)
        b_q, b_k, b_v = mha.in_proj_bias.split(dim, dim=0)
        n_latents = queries.size(1)
        q = F.linear(self.query_norm(queries), w_q, b_q).view(bsz, n_latents, heads, head_dim).transpose(1, 2)

        def chunk_contrib(x_chunk, q_):
            ctx = self.context_norm(self.input_proj(self.input_norm(x_chunk)))
            k = F.linear(ctx, w_k, b_k).view(bsz, -1, heads, head_dim).transpose(1, 2)
            v = F.linear(ctx, w_v, b_v).view(bsz, -1, heads, head_dim).transpose(1, 2)
            scores = torch.matmul(q_, k.transpose(-2, -1)) * scale          # [B,H,Lq,c]
            chunk_max = scores.amax(dim=-1)                                 # [B,H,Lq]
            weights = torch.exp(scores - chunk_max.unsqueeze(-1))
            denom = weights.sum(dim=-1)                                     # [B,H,Lq]
            out = torch.matmul(weights, v)                                  # [B,H,Lq,Dh]
            return chunk_max, denom, out

        run_max = run_l = run_o = None
        for start in range(0, n_sentences, chunk_size):
            x_chunk = x[:, start:start + chunk_size, :]
            if self.training and torch.is_grad_enabled():
                c_max, c_l, c_o = checkpoint(chunk_contrib, x_chunk, q, use_reentrant=False)
            else:
                c_max, c_l, c_o = chunk_contrib(x_chunk, q)
            if run_max is None:
                run_max, run_l, run_o = c_max, c_l, c_o
            else:
                new_max = torch.maximum(run_max, c_max)
                alpha = torch.exp(run_max - new_max)
                beta = torch.exp(c_max - new_max)
                run_l = alpha * run_l + beta * c_l
                run_o = alpha.unsqueeze(-1) * run_o + beta.unsqueeze(-1) * c_o
                run_max = new_max
        attended = (run_o / run_l.unsqueeze(-1)).transpose(1, 2).reshape(bsz, n_latents, dim)
        return F.linear(attended, mha.out_proj.weight, mha.out_proj.bias)

    def forward_stem(self, x, key_padding_mask=None, chunk_size=None):
        if chunk_size is None:
            chunk_size = getattr(self, "_stem_chunk_size", None)
        queries = self.latent_dropout(self.latent_array).unsqueeze(0).expand(x.size(0), -1, -1)
        if chunk_size and key_padding_mask is None and x.size(1) > int(chunk_size):
            attended = self._cross_attend_chunked(x, queries, int(chunk_size))
        else:
            context = self.context_norm(self.input_proj(self.input_norm(x)))
            attended, _ = self.cross_attention(
                query=self.query_norm(queries),
                key=context,
                value=context,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        latents = queries + self.dropout(attended)
        return self.output_norm(self.self_attention(latents))

    def forward_tail(self, latents):
        for stage in self.reduction_stages:
            latents = stage(latents)
        return latents

    def forward(self, x, key_padding_mask=None, chunk_size=None):
        return self.forward_tail(self.forward_stem(x, key_padding_mask=key_padding_mask, chunk_size=chunk_size))


class GameCentroidExpander(nn.Module):
    """Projection head used only by the game-level VICReg regularizer.

    Input/output:
        centroid: (batch, output_dim), default (batch, 18)
        expanded: (batch, expander_dim), default (batch, 1024)

    The final layer is raw: VICReg's variance/covariance terms own the scale and
    decorrelation pressure.
    """

    def __init__(
        self,
        input_dim=18,
        hidden_dims=(128, 512),
        output_dim=1024,
        dropout=0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.output_dim = int(output_dim)
        self.input_norm = nn.LayerNorm(self.input_dim)

        dims = [self.input_dim, *self.hidden_dims, self.output_dim]
        layers = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, centroid):
        if centroid.dim() != 2:
            raise ValueError(f"GameCentroidExpander expects (batch, dim), got {tuple(centroid.shape)}.")
        return self.net(self.input_norm(centroid))


def game_centroid(latents):
    """Mean-pool latent slots into one compact vector per game."""
    latents = latents.float()
    if latents.dim() == 3:
        return latents.mean(dim=1)
    if latents.dim() == 2:
        return latents
    raise ValueError(f"Expected latents with shape (B,L,D) or centroids (B,D), got {tuple(latents.shape)}.")


class TagRegressionHead(nn.Module):
    """Validation-only probe: map a frozen encoder code to tag signals.

    Input is the encoder output (batch, num_latents, latent_out_dim). By default
    the latent set is flattened (num_latents * latent_out_dim) so the head sees
    the full representation; use pool="mean" to average over latents instead.
    pool="stats" concatenates mean/std/max/min over latent slots, which gives a
    compact probe for small validation sets.

    Outputs a dict with raw presence logits and raw-count regression logits.
    Apply sigmoid() to presence_logits for tag-existence probabilities. Apply
    softplus() and expm1() to count_logits to recover non-negative tag counts.
    This head is never part of the VICReg loss path -- it is trained separately
    on a frozen encoder.
    """

    def __init__(
        self,
        num_tags,
        num_latents=256,
        latent_out_dim=18,
        hidden_dims=(256, 128),
        dropout=0.1,
        pool="flatten",
    ):
        super().__init__()
        if pool not in ("flatten", "mean", "stats"):
            raise ValueError("pool must be 'flatten', 'mean', or 'stats'.")
        self.pool = pool
        self.num_tags = int(num_tags)
        if pool == "flatten":
            in_dim = num_latents * latent_out_dim
        elif pool == "stats":
            in_dim = latent_out_dim * 4
        else:
            in_dim = latent_out_dim
        layers = []
        prev = in_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.LayerNorm(prev),
                nn.Linear(prev, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = hidden_dim
        layers += [nn.LayerNorm(prev)]
        self.trunk = nn.Sequential(*layers)
        self.presence = nn.Linear(prev, self.num_tags)
        self.count = nn.Linear(prev, self.num_tags)

    def forward(self, feats):
        if self.pool == "flatten":
            x = feats.flatten(start_dim=1)
        elif self.pool == "stats":
            x = torch.cat(
                [
                    feats.mean(dim=1),
                    feats.std(dim=1, unbiased=False),
                    feats.amax(dim=1),
                    feats.amin(dim=1),
                ],
                dim=1,
            )
        else:
            x = feats.mean(dim=1)
        x = self.trunk(x)
        return {
            "presence_logits": self.presence(x),
            "count_logits": self.count(x),
        }


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
    treat all latent vectors in the batch as the sample axis, so the covariance
    matrix is output_dim x output_dim (18 x 18 by default), not the flattened
    num_latents x output_dim feature set.
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


def vicreg_centroid_loss(
    centroid_a,
    centroid_b,
    expander,
    invariance_weight=25.0,
    variance_weight=25.0,
    covariance_weight=1.0,
    compact_variance_weight=0.0,
    compact_covariance_weight=0.0,
    eps=1e-4,
):
    """VICReg loss where the sample axis is games, not latent slots.

    Invariance is the MSE between the compact game centroids. Variance and
    covariance are computed on expander(centroid), so the regularizer directly
    spreads different games apart and has enough dimensions to do it cleanly.
    """

    if centroid_a.shape != centroid_b.shape:
        raise ValueError(
            f"VICReg centroid views must match, got {centroid_a.shape} and {centroid_b.shape}."
        )
    if centroid_a.dim() != 2:
        raise ValueError(f"VICReg centroid loss expects (batch, dim), got {tuple(centroid_a.shape)}.")

    centroid_a = centroid_a.float()
    centroid_b = centroid_b.float()
    repr_loss = F.mse_loss(centroid_a, centroid_b)
    reg_a = expander(centroid_a).float()
    reg_b = expander(centroid_b).float()

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

    std_loss = 0.5 * (variance_term(reg_a) + variance_term(reg_b))
    cov_loss = 0.5 * (covariance_term(reg_a) + covariance_term(reg_b))
    compact_std_loss = 0.5 * (variance_term(centroid_a) + variance_term(centroid_b))
    compact_cov_loss = 0.5 * (covariance_term(centroid_a) + covariance_term(centroid_b))
    total = (
        invariance_weight * repr_loss
        + variance_weight * std_loss
        + covariance_weight * cov_loss
        + compact_variance_weight * compact_std_loss
        + compact_covariance_weight * compact_cov_loss
    )
    return {
        "loss": total,
        "invariance": repr_loss,
        "variance": std_loss,
        "covariance": cov_loss,
        "compact_variance": compact_std_loss,
        "compact_covariance": compact_cov_loss,
    }


class SentimentAdversarialLoss(nn.Module):
    """GRL loss that pushes latent vectors toward SST-head uncertainty.

    The SST head is a frozen 0..1 regressor. We use Bernoulli entropy as a
    confidence surrogate. Minimizing entropy after a GRL makes the encoder ascend
    that entropy, so the frozen sentiment head is driven toward uncertainty.
    """

    def __init__(
        self,
        sentiment_head,
        input_dim=18,
        probe_hidden=256,
        probe_dim=1024,
        grl_lambda=1.0,
        eps=1e-6,
        normalize=True,
    ):
        super().__init__()
        self.sentiment_head = sentiment_head
        self.grl = GradientReversal(grl_lambda)
        # Learnable up-projection probe, placed AFTER the GRL: it tries to recover
        # sentiment confidence from the compact encoder code, while the GRL makes
        # the encoder fight it. Its own gradients are NOT reversed. Biases are
        # disabled so the probe cannot use a learned constant channel shortcut.
        self.probe = nn.Sequential(
            nn.Linear(input_dim, probe_hidden, bias=False),
            nn.GELU(),
            nn.Linear(probe_hidden, probe_dim, bias=False),
        )
        self.eps = eps
        self.normalize = normalize

    def forward(self, latents):
        # Run the probe + frozen head + entropy in fp32: normalize() and the
        # Bernoulli entropy overflow to NaN easily under AMP fp16.
        with torch.amp.autocast("cuda", enabled=False):
            flat = latents.reshape(-1, latents.size(-1)).float()
            up = self.probe(self.grl(flat))
            if self.normalize:
                up = F.normalize(up, p=2, dim=-1)
            pred = self.sentiment_head(up).float().clamp(self.eps, 1.0 - self.eps)
            entropy = -(pred * pred.log() + (1.0 - pred) * (1.0 - pred).log())
            loss = entropy.mean()
            with torch.no_grad():
                stats = {
                    "sentiment_mean": pred.mean(),
                    "sentiment_std": pred.std(unbiased=False),
                    "sentiment_entropy": entropy.mean(),
                }
        return loss, stats
