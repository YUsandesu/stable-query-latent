# Model Comparison — v1 vs v2 vs base

Comparison of three architectures on the PXI multi-variant benchmark:

- **v1** — original `LatentQueryFlatRegressor` (one learnable latent array per stage).
- **v2** — `LatentQueryFunnelRegressor` (one learnable array + self-attention, later
  queries are linear reductions feeding a cross-attention funnel; see
  [`README_2.md`](README_2.md)).
- **base** — `LatentQueryBaseRegressor`: only the first latent array + self-attention,
  then each latent's **feature dim** is funneled `64→16→8` and projected to output.
  No cross-attention funnel. The smallest model.

## Setup

- **Data:** `pesudo_data/benchmark_sentence_latent_query_multi.h5`
  (1805 samples; train 1475 / test 330).
- **Task:** per-dimension 5-class classification over 10 score dimensions.
- **Split:** `--split-by score_combo` (test groups held out by full target combo, so
  absolute numbers shift between seeds — within a seed, v1/v2/control share the
  identical split, so per-seed differences are a valid **paired** test).
- **Identical for every run:** epochs=800, batch=128, lr=3e-4 → cosine → 1e-5,
  weight_decay=1e-4, flat_dim=128, query_sizes=(32,16,8), num_heads=8, dropout=0.0.
  The **only** things that change are `--model` and (for the control) `--hidden-dim`.
  Same training loop (shared `train_and_test`), same seed.
- Checkpoint selection: best **test_mae** epoch.

## Results (10 seeds, mean ± std)

Aggregated over **10 random seeds** `[412384, 701183, 792180, 283149, 493976, 27630,
69219, 415472, 273118, 633492]` (generated from meta-seed 20240622). Each config
trained 800 epochs per seed. Raw per-run numbers in
[`heads/seed_sweep.json`](heads/seed_sweep.json) (v1/v2) and
[`heads/seed_sweep_base.json`](heads/seed_sweep_base.json) (base).

| model | params | test_acc ↑ | test_mae ↓ | test_ce ↓ |
|---|---:|---:|---:|---:|
| v1 (hidden64)   | 292,914 | 0.9545 ± 0.0069 | 0.0634 ± 0.0073 | 0.2063 ± 0.0331 |
| v2 (hidden64)   | 342,026 | 0.9765 ± 0.0056 | 0.0330 ± 0.0081 | 0.1009 ± 0.0297 |
| v2 (hidden56)   | 278,706 | 0.9802 ± 0.0068 | 0.0285 ± 0.0090 | 0.0803 ± 0.0257 |
| **base (hidden64)** | **182,442** | **0.9865 ± 0.0049** | **0.0234 ± 0.0094** | **0.0482 ± 0.0203** |

### Paired differences (same split per seed)

| comparison | Δacc (mean ± std) | Δmae | acc win-rate |
|---|---:|---:|---:|
| v2(h64) − v1(h64)   | +0.0220 ± 0.0061 | −0.0304 | 10/10 |
| v2(h56) − v1(h64)   | +0.0257 ± 0.0057 | −0.0349 | 10/10 |
| v2(h56) − v2(h64)   | +0.0037 ± 0.0066 | −0.0045 | 7/10 |
| **base − v1(h64)**  | **+0.0321 ± 0.0051** | −0.0400 | **10/10** |
| **base − v2(h64)**  | **+0.0101 ± 0.0044** | −0.0096 | **10/10** |
| **base − v2(h56)**  | **+0.0064 ± 0.0040** | −0.0051 | **9/10** |

## Findings

1. **base is the best — and the smallest.** With **182k params (≈half of v2(h64))**
   it tops every metric. Paired vs v2(h64): Δacc **+0.0101 ± 0.0044, 10/10** (effect
   ~2.3× its std → significant). Paired vs the previous best v2(h56): Δacc
   **+0.0064 ± 0.0040, 9/10** (~1.6× std → robust). So **the cross-attention funnel in
   v2 is a net negative** here — dropping it and instead funneling the *feature* dim
   (64→16→8) in the head both shrinks the model and improves it.
2. **"Shrink params → better generalization" holds on this dataset.** Ranked by acc,
   the order is **base (182k) > v2 h56 (279k) > v2 h64 (342k) > v1 (293k)** — the
   smallest model wins. Aggressive feature bottleneck + keeping all 32 latents +
   self-attention is a stronger inductive bias than a deeper cross-attention stack on
   only 1475 training samples. (Same lesson as SST, where the narrow `64→16→4` head won.)
3. **v2 still clearly beats v1** (Δacc +2.2–2.6%, 10/10), so the *first* changes —
   single learnable array + self-attention — are real wins. It's the *extra
   cross-attention funnel stages* that don't pay off; the feature-funnel head captures
   the benefit more cheaply.
4. **hidden56 vs hidden64 (within v2) is a wash** (Δacc +0.0037, std 0.0066, 7/10) —
   within noise; consistent with the "smaller is at least as good" trend.

## Interpretation

What actually helps, in order of impact:

1. **One learnable latent array + self-attention over the latents** — the big jump
   (v1 → v2 → base all keep this). Self-attention lets the 32 latents integrate
   globally before the head.
2. **An aggressive bottleneck before the output** — base funnels each latent's feature
   dim 64→16→8; this regularizes hard on the small dataset and is cheaper than v2's
   latent-count cross-attention funnel.
3. **The cross-attention funnel (v2's reduce→cross stages) is *not* worth it** — it
   adds parameters and slightly hurts vs the plain feature-funnel head.

## Reproduce

```bash
# 10-seed sweep, all three configs (sequential; writes heads/seed_sweep.json)
python PXIbench_test/seed_sweep.py

# same sweep but several configs run concurrently on one GPU (much faster)
python PXIbench_test/seed_sweep_parallel.py --workers 3

# base sweep (10 seeds, parallel) -> heads/seed_sweep_base.json
python PXIbench_test/seed_sweep_parallel.py --models base_h64 --workers 3 --out seed_sweep_base.json

# a single config/seed by hand (paths resolve relative to PXIbench_test/, so use heads/...)
python PXIbench_test/test_latent_query_model.py --model v1   --epochs 800 --seed 42
python PXIbench_test/test_latent_query_model.py --model v2   --epochs 800 --seed 42
python PXIbench_test/test_latent_query_model.py --model base --epochs 800 --seed 42
```

## Caveats

- The `score_combo` split makes absolute numbers swing between seeds (the per-seed
  difficulty of the held-out combos varies). Compare same-seed pairs / the 10-seed
  aggregate, not raw numbers across seeds.
- v2 trains ~15% slower per wall clock (≈1m46s vs 1m30s at hidden64) due to the extra
  self-attention block — a small cost for the accuracy gain. On this tiny model the
  GPU sits at ~44% utilization (overhead-bound); `seed_sweep_parallel.py` packs
  several runs onto the card to recover throughput.
