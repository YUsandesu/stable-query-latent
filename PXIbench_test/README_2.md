# Latent Query Model — v2 (Funnel) Architecture

A redesign of the original `LatentQueryFlatRegressor` (see [`README.md`](README.md))
that keeps **only the first learnable latent array** and derives every later query
by linearly compressing the previous stage's latents, with a self-attention layer
added after the first cross-attention.

- Model: [`latent_query_model_v2.py`](../latent_query_model_v2.py) → `LatentQueryFunnelRegressor`
- Trainer: same as v1, selected with `--model v2`
  ([`test_latent_query_model.py`](test_latent_query_model.py)), or the thin entry
  point [`test_latent_query_model_v2.py`](test_latent_query_model_v2.py).
- Constructor signature is **identical** to `LatentQueryFlatRegressor`, so it is a
  drop-in swap.

## Motivation

In v1 every stage owns an independent learnable query array (`nn.Parameter`).
Those queries are **static** — they do not depend on the current sample until they
attend. v2 makes the queries **input-adaptive**: after the first stage aggregates
the sentence sequence into a latent set, each subsequent query is a learned linear
combination of the latents the model has *already* produced for *this* sample.

## Architecture

Default `query_sizes=(32, 16, 8)`, `hidden_dim=D`:

```
input (B, N, input_dim)
  │  Linear(input_dim → D)
  ▼
stage 0  ── LEARNABLE latent array [32]  ──cross-attend──▶ projected input   ⇒ L0 (B,32,D)
            self-attention(L0)                                                ⇒ L0 (B,32,D)   ← NEW
  │
stage 1  ── LatentReducer Linear 32→16 over latent axis   ⇒ Q1 (B,16,D)
            Q1 ──cross-attend──▶ L0  (the first latent array)                 ⇒ L1 (B,16,D)
  │
stage 2  ── LatentReducer Linear 16→8 over latent axis    ⇒ Q2 (B,8,D)
            Q2 ──cross-attend──▶ L1  (previous cross-attention)               ⇒ L2 (B,8,D)
  │
  ▼
head: LayerNorm → Linear(8·D → flat_dim) → GELU → Linear(flat_dim → output_dim)
```

### What changed vs v1

| | v1 (`LatentQueryFlatRegressor`) | v2 (`LatentQueryFunnelRegressor`) |
|---|---|---|
| Learnable latent arrays | one per stage (3 total) | **only stage 0** |
| Later-stage queries | independent learnable `nn.Parameter` | **`LatentReducer`**: Linear over the latent axis of the previous stage's output |
| Self-attention | none | **one block after the first cross-attention** |
| Cross-attention context | each stage attends to the previous stage's latents | same (stage 1 → L0, stage 2 → L1) |
| Head | flatten(last latents) → MLP | unchanged |

### Key building blocks (in `latent_query_model_v2.py`)

- **`LatentArrayCrossAttention`** — cross-attention with a learnable query array
  (used only at stage 0; identical to v1's block).
- **`QueryCrossAttention`** — cross-attention whose query is supplied externally
  (used at every later stage; no learnable query of its own).
- **`SelfAttention`** — pre-norm self-attention block over the latent set.
- **`LatentReducer`** — `(B, n_in, D) → (B, n_out, D)` via a `Linear(n_in → n_out)`
  applied on the transposed latent axis. Each output query is a learned linear
  mix of the input latents.

All blocks are pre-norm with residual connections and a 4× GELU FFN, matching v1.

## Generalization

The design generalizes to any `query_sizes` of length ≥ 1:
- `query_sizes[0]` → the single learnable latent array (+ self-attention).
- each subsequent size → one `LatentReducer` + one `QueryCrossAttention` stage that
  queries the previous stage's latents.

So `query_sizes=(32, 16, 8, 4)` gives stage 0 (32, learnable) → reduce→16 → reduce→8
→ reduce→4, all cross-attending to the prior stage.

## Usage

```bash
# Train v2 (defaults to the multi dataset under pesudo_data/)
python PXIbench_test/test_latent_query_model_v2.py

# …or via the shared trainer's flag
python PXIbench_test/test_latent_query_model.py --model v2

# v1 is still the default
python PXIbench_test/test_latent_query_model.py            # == --model v1
```

The checkpoint records `model_name` and `param_count` alongside the usual config,
so a saved `.pt` always knows which architecture produced it.

## Results

v2 beats v1 on every metric across two seeds — **even when shrunk to fewer
parameters than v1**, so the gain is architectural, not capacity. Full numbers and
methodology in [`COMPARISON.md`](COMPARISON.md).
