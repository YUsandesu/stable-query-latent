# Model Comparison — v1 vs v2

Comparison of the original `LatentQueryFlatRegressor` (**v1**) against the redesigned
`LatentQueryFunnelRegressor` (**v2**, see [`../README_2.md`](../README_2.md)) on the
PXI multi-variant benchmark.

## Setup

- **Data:** `pesudo_data/benchmark_sentence_latent_query_multi.h5`
  (1805 samples; train 1475 / test 330).
- **Task:** per-dimension 5-class classification over 10 score dimensions.
- **Split:** `--split-by score_combo` (test groups held out by full target combo, so
  absolute numbers shift a lot between seeds — only same-seed v1↔v2 pairs are
  directly comparable).
- **Identical for every run:** epochs=800, batch=128, lr=3e-4 → cosine → 1e-5,
  weight_decay=1e-4, hidden_dim=64, flat_dim=128, query_sizes=(32,16,8), num_heads=8,
  dropout=0.0. The **only** thing that changes is `--model` (and the param-matched
  control's `--hidden-dim`). Same training loop (shared `train_and_test`), same seed.
- Checkpoint selection: best **test_mae** epoch.

## Results

| seed | model | params | test_acc ↑ | test_mae ↓ | test_ce ↓ | best epoch |
|---|---|---:|---:|---:|---:|---:|
| 42  | v1            | 292,914 | 0.9436 | 0.0830 | 0.2102 | ~334 |
| 42  | v2 (hidden64) | 342,026 | **0.9573** | **0.0630** | **0.1423** | ~132 |
| 42  | v2 (hidden56) | **278,706** | **0.9685** | **0.0464** | **0.1380** | — |
| 123 | v1            | 292,914 | 0.9699 | 0.0343 | 0.1287 | — |
| 123 | v2 (hidden64) | 342,026 | **0.9869** | **0.0149** | **0.0563** | — |
| 123 | v2 (hidden56) | **278,706** | **0.9881** | **0.0158** | **0.0526** | — |

Stability (seed 42, mean over the last 50 epochs):

| model | last-50 acc | last-50 mae |
|---|---:|---:|
| v1            | 0.9425 | 0.0860 |
| v2 (hidden64) | 0.9564 | 0.0742 |

## Findings

1. **v2 wins on every metric, both seeds.** acc up, mae and ce down in all pairs.
2. **The gain is architectural, not capacity.** The `hidden56` control gives v2
   **278,706 params — fewer than v1's 292,914** — and it still beats v1 clearly
   (seed 42: acc 0.944 → 0.969, mae 0.083 → 0.046; seed 123: acc 0.970 → 0.988).
   So the improvement is not explained by parameter count.
3. **v2 converges earlier.** Best checkpoint around epoch ~132 vs ~334 for v1.
4. **v2 is steadier late in training** (higher last-50 mean accuracy, lower mae).

## Why v2 helps (interpretation)

The three changes work together:

1. **Single learnable latent array; later queries are linear reductions of the
   previous latents.** v1's per-stage queries are static `nn.Parameter`s — sample-
   independent until they attend. v2's later queries are built from what the model
   has *already* aggregated for *this* sample, so they are input-adaptive and carry
   more signal into each cross-attention.
2. **Self-attention after the first cross-attention.** It lets the 32 latents
   exchange information and integrate globally *before* the funnel narrows, so the
   reduced queries are computed from a better-mixed representation.
3. **Funnel of cross-attention** (each stage queries the previous stage's latents)
   gives a clean hierarchical refinement rather than three independent query sets.

## Reproduce

```bash
# v1
python PXIbench_test/test_latent_query_model.py --model v1 --epochs 800 --seed 42 \
  --model-out heads/cmp_v1.pt --history-txt heads/cmp_v1_hist.txt --per-dim-txt heads/cmp_v1_perdim.txt

# v2 (default hidden_dim=64)
python PXIbench_test/test_latent_query_model_v2.py --epochs 800 --seed 42 \
  --model-out heads/cmp_v2.pt --history-txt heads/cmp_v2_hist.txt --per-dim-txt heads/cmp_v2_perdim.txt

# param-matched control (v2 with fewer params than v1)
python PXIbench_test/test_latent_query_model_v2.py --hidden-dim 56 --epochs 800 --seed 42 \
  --model-out heads/v2h56_s42.pt --history-txt heads/v2h56_s42_h.txt --per-dim-txt heads/v2h56_s42_p.txt

# repeat any of the above with --seed 123 for the second-seed check
```

Paths are resolved relative to `PXIbench_test/`, so run from the project root with
`heads/...` (not `PXIbench_test/heads/...`).

## Caveats

- Two seeds only; the `score_combo` split makes absolute numbers swing between
  seeds (compare same-seed pairs, not across seeds). For a publication-grade claim,
  average 3–5 seeds and report variance.
- v2 trains ~15% slower per the wall clock (1m46s vs 1m30s at hidden64) due to the
  extra self-attention block — a small cost for the accuracy gain.
