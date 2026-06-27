# Tag-probe results & limitation analysis

Goal set by the task: push the tag-prediction **micro-F1 to ≥ 0.85**, allowed to
(a) strengthen the *siamese* encoder (more latent arrays, deeper encoder — but
keep the VICReg/IVCReg + Latent-Array training core) and (b) redesign the probe
with a fairer evaluation (cross-validation; only score tags that appear in the
train split ≥ 2 times).

**Verdict: 0.85 is not reachable on this dataset, and the bottleneck is the data,
not the encoder or the probe.** Details below, all numbers are 5-fold CV with the
fairness rule and the decision threshold tuned on train only (no val leakage).

## 2026-06-25 final update: hierarchical self-attention + description alignment + recommendation decorrelation

The latest checkpoint is `heads/hierarchical64_align_reco/vicreg_review_h5_latest.pt`.
It keeps the centroid-level VICReg repair, but adds hierarchical latent self-attention,
full-text description alignment, and a game-level recommendation-rate decorrelation term.

| Diagnostic | Result |
|---|---:|
| compact centroid Participation Ratio | 26.67 |
| z-scored compact centroid PR | 26.64 |
| Cyberpunk / AO diagnostic text ranks | 1 / 1 |
| TAP tag micro-F1, flatten pool | 0.6938 |
| content retention vs raw | 0.888 |
| sentiment R² retention vs raw | 0.349 |
| recommendation-rate probe Pearson (CV / holdout) | -0.068 / 0.089 |

Interpretation:

- The tag probe is still healthy, but the stronger result is selectivity.
- Content survives; sentiment and recommendation-rate linear readouts are both strongly reduced.
- The identity retrieval success on the diagnostic text set is now rank-1 across Cyberpunk
  and Across the Obelisk, but that includes the long-text cases used in the
  alignment cache, so it is best read as a trained alignment success rather than pure
  zero-shot retrieval.
- Historical sections below describe earlier checkpoint families (`centroid64_grl` and
  the preceding pure-VICReg ablations).

## 2026-06-25 update: game-centroid VICReg fixes low-rank collapse, with mixed identity retrieval

The low-rank failure was caused by computing VICReg variance/covariance on
`(batch * latent_slots, output_dim)`: within-game slot spread satisfied the
regularizer while game centroids stayed collapsed. The H5 trainer now supports
`--vicreg-scope game`, which mean-pools slots before the loss, applies
invariance on compact game centroids, and applies variance/covariance on a
512-d expander projection. The deployed repair uses a 64-d compact centroid
(`--output-dim 64`) plus a short GRL fine-tune.

Final checkpoint:
`VICReg_review/heads/centroid64_grl/vicreg_review_h5_latest.pt`.

| Diagnostic | Previous collapsed 18-d code | New 64-d game-centroid code |
|---|---:|---:|
| compact game-vector Participation Ratio | ~2 (failure case) | **15.97** |
| Cyberpunk 2077 description rank | old Cyberpunk text ranks 273-288 | **25 / 293** |
| Cyberpunk 2077 neutral / positive / negative ranks | 288 / 273 / 284 | **41 / 41 / 12** |
| Across the Obelisk neutral / positive / negative ranks | 1 / 8 / 24 | 26 / 27 / 29 (**regression**) |
| TAP tag micro-F1, flatten pool | 0.435 old Steam-tag run / 0.448 content run | **0.686** |
| TAP tag micro-F1, stats pool export | 0.394 old Steam-tag run | **0.687** |
| content retention vs raw | 0.844 | **0.951** |
| sentiment R² retention vs raw | 0.682 | 0.835 |

Notes:

- The new rank diagnostic uses z-scored cosine over 293 training-game centroids.
  Raw Qwen mean-pool PR on the same cache is 18.27, so the repaired VICReg
  centroid reaches the same order of effective dimension while remaining compact.
- TAP labels in the current H5 are 23 coarse non-subjective labels, so the
  selectivity probe reports `subjective=nan`; the meaningful checks are content
  retention and SST sentiment R² retention.
- The final checkpoint improves the collapsed Cyberpunk-style failure mode
  and restores compact centroid PR, while keeping tag F1 essentially unchanged.
  It is **not** universally better for identity retrieval: Across the Obelisk
  regresses from the old flattened-identity ranks `1 / 8 / 24` to `26 / 27 / 29`
  on the compact centroid. Sentiment remains suppressed relative to raw
  (`0.835` retention), though the content-sentiment gap is smaller than the old
  18-d pure compression ablation. The result is a better PR/Cyberpunk-collapse
  repair, not a finished nearest-neighbor identity encoder.

## The dataset is the binding constraint

- **293 games total, 283 labeled.** That is the entire corpus. Per fold ≈ 234
  train games.
- **219 Steam tags**, long-tailed: median tag appears in **12** games, 79 tags in
  fewer than 10. Steam caps tags at 20/game, so the positive/negative boundary
  for mid-frequency tags is a fuzzy crowd-vote ranking, not a clean label.
- A tag with 12 positives has ~10 training examples per fold. No classifier
  generalizes a 219-way multilabel problem from that.

## The ceiling experiment (the decisive result)

`ceiling_diagnostic.py` runs the *same* CV pipeline on the **raw 1024-d Qwen
sentence embeddings** (mean-pooled per game). Because the VICReg code is a lossy
compression of exactly these embeddings, raw-embedding F1 is a hard **upper
bound** on any frozen-encoder probe.

| Representation | full 219 tags | freq≥40 (37 tags) | freq≥80 (12 tags) |
|---|---|---|---|
| **Raw Qwen embedding — per-tag logreg** (CEILING) | **0.504** | 0.605 | 0.703 |
| Raw Qwen embedding — shared nonlinear MLP | 0.540 | 0.633 | 0.718 |
| Frozen VICReg code, flatten pool (epoch 39) | 0.435 | 0.559 | 0.677 |
| Frozen VICReg code, stats pool (epoch 39) | 0.394 | 0.550 | 0.678 |

A nonlinear MLP on the raw embeddings (0.540) barely beats the linear probe
(0.504): the limit is the **information in the data**, not classifier capacity.
Even cherry-picking the 12 most frequent tags, the ceiling is ~0.70.

## Coarser labels don't rescue it

The "labels are too discrete" intuition is correct, but coarsening only helps so
much — and the high numbers come from near-universal labels, not real signal:

| Label set | # labels | raw ceiling micro-F1 | densest subset |
|---|---|---|---|
| Steam tags | 219 | 0.50 | 0.70 (12 tags) |
| Genres | 12 | 0.68 | 0.73 |
| Categories | 26 | 0.72 | 0.82 (9 tags) |

The categories "freq≥80" subset (0.82) is dominated by labels like *Single-player*
that are present on almost every game — high micro-F1 there is trivial recall, not
a meaningful diagnostic (its macro-F1 is only 0.52).

## What the redesigned probe changed

`train_tag_probe.py` was rewritten (the old version used a single random 80/20
split, which on ~290 games swings several F1 points by luck):

- **5-fold cross-validation**, reporting per-fold mean ± std.
- **Fairness rule**: a tag is scored on a fold only if it has ≥ `--min-train-pos`
  (default 2) positives in that fold's train split and ≥ 1 in val. Tags that are
  unique to one game — the exact case the task flagged — are excluded from the
  metric instead of silently zeroing it.
- **Threshold tuned on train only**, per tag.
- **Frequency-floor breakdown** so the discreteness effect is visible directly.
- `--pool flatten|stats|mean` (flatten = full code, scores highest).

## Where the encoder actually stands

The frozen VICReg code reaches **0.435** (flatten) vs the **0.504** raw ceiling —
a ~0.07 gap. That gap is real and partly recoverable, because the encoder throws
away tag-relevant information through:

1. **An 18-d output bottleneck** per latent (1024→…→18). VICReg's covariance term
   is only `output_dim × output_dim`, so widening to 64–128 is cheap and safe.
2. **The GRL sentiment adversary**, which is *designed* to destroy a direction of
   variance; some of that variance correlates with tags.

Restoring the deeper encoder (the self-attention `blocks` + `cross_mlp` variant
saved in `heads/vicreg_review_h5_best.pt`) and widening `--output-dim` would move
the probe from ~0.435 toward the ~0.50 ceiling. **It cannot exceed ~0.50**, so it
does not change the verdict.

## How to actually get a high F1 (changing the data, not the model)

Per `sst/heads/RESULTS.md`'s lesson — "to break the ceiling, change the embedding,
not the head" — the analogue here is: **change the data, not the encoder.**

1. **More games.** 293 is the core problem. A few thousand labeled games would let
   the tail tags become learnable. This is the single highest-leverage change.
2. **Curate a content-bearing tag set.** Drop technical/meta/demographic tags
   (*Controller, Early Access, Great Soundtrack, 1980s, Female Protagonist*) that
   reviews don't describe; keep genre/mechanic tags (*RPG, Racing, Horror, Visual
   Novel, Deckbuilding*) that already score 0.75–0.89 per-tag. A focused ~30-tag
   "predict the genre from reviews" task is honest and lands ~0.65–0.75.
3. Only then is encoder tuning worth the GPU time.

## Adversary-weight sweep — a tuned adversary DOES help (final result)

All epoch 39, seed 42, identical config except `--adversary-weight` (GRL schedule
warmup 5 / ramp 10 / lambda 1). Selectivity = content_retention − sentiment_retention.

| adv weight | content ret | subjective ret | sentiment ret | **selectivity gap** |
|---|---|---|---|---|
| 0 (pure VICReg) | 0.820 | 0.792 | 0.486 | +0.335 |
| 1 (original)    | 0.844 | 0.789 | 0.682 | +0.162 |
| **10** | **0.850** | 0.831 | **0.411** | **+0.439** |
| 20 | 0.836 | 0.881 | 0.610 | +0.226 |
| 50 | 0.785 | 0.776 | 0.639 | +0.146 |

**Conclusion — the hypothesis holds when the adversary is properly weighted.** At
`--adversary-weight 10` the sentiment adversary *beats* pure VICReg on both axes:
content retention rises to **0.85** (the highest of any run) while sentiment
retention falls to **0.41** (sentiment R² 0.90→0.37) — the largest selectivity gap,
**+0.439**. So yes, the siamese model + sentiment adversary successfully extracts
[mechanics + story] while filtering subjective opinion — but only in a tuned band:
weight 1 is too weak to matter, and weights ≥20 over-regularize and *hurt* (the
encoder finds a degenerate solution that leaks more sentiment again). The
relationship is non-monotonic; weight ~10 is the sweet spot here.

The deployable probe (`heads/tag_probe_linear.pt`, used by `validation.py`) is
exported from the weight-10 encoder.

## The right evaluation for the actual hypothesis (selectivity)

The hypothesis isn't "tag-F1 ≥ 0.85" — it's "the siamese encoder + sentiment
adversary **keeps story/mechanics, drops subjective opinion**." That's a
*relative* claim, so the fair test is relative. `probe_selectivity.py` measures,
under the same fair CV, three decodabilities on the raw embedding (ceiling) and on
the VICReg code, then reports **retention = vicreg / raw**:

| Axis | raw (ceiling) | VICReg code | retention |
|---|---|---|---|
| content F1 (mechanics+story, 133 tags) | 0.531 | 0.448 | **0.844** |
| subjective F1 (affect/quality, 30 tags) | 0.374 | 0.295 | 0.789 |
| sentiment R² (mean SST score) | 0.904 | 0.617 | **0.682** |

**Selectivity gap = content_retention − sentiment_retention = +0.16.** The
ordering `content (0.84) > subjective (0.79) > sentiment (0.68)` matches the
hypothesis: content is retained best, the sentiment axis the GRL attacks is
suppressed most. The tag partition lives in `tag_groups.py` → `tags/tag_groups.json`
(mechanics / story / subjective / drop) and is meant to be edited.

**Two honest caveats:**

1. **The effect is modest** (sentiment is only ~30% suppressed, not erased). The
   current checkpoint is epoch 39 with a 5-epoch GRL warmup + 10-epoch ramp, so
   full-strength GRL has only run ~24 epochs. Pushing `--grl-lambda`,
   `--adversary-weight`, and training longer is the legitimate "improve the
   siamese model" direction now — and `probe_selectivity.py` is the metric to
   optimize, replacing the tag-F1 chase.
2. **Attribution needs an ablation.** The content>sentiment gap could come from
   the adversary *or* just from VICReg's 18-d compression happening to hurt
   sentiment more. The clean test: train one checkpoint with `--adversary-weight 0`
   (pure VICReg, no GRL) and re-run `probe_selectivity.py`. If the selectivity gap
   collapses without the adversary, the adversary is doing the work. **This is the
   single most important next run.** The sentiment target is the same SST head the
   adversary fools, which is appropriate (it's exactly the axis the adversary
   optimizes against), but means raw sentiment R² is near-ceiling by construction.

## Ablation: is it the adversary, or just VICReg? (the decisive run)

Two checkpoints, both epoch 39, seed 42, identical config except
`--adversary-weight` (1 vs 0). Selectivity probe on each, flatten pool.

**Retention (vicreg / raw ceiling), higher = signal kept:**

| Axis | Pure VICReg (no adversary) | VICReg + sentiment adversary |
|---|---|---|
| content (mechanics+story) | 0.820 | 0.844 |
| subjective (affect/quality) | 0.792 | 0.789 |
| sentiment (SST R²) | **0.486** | **0.682** |
| **selectivity gap (content − sentiment)** | **+0.335** | +0.162 |

Absolute (micro-F1 / R²): raw ceiling content=0.531 subj=0.374 sent_R²=0.904;
pure-VICReg 0.435 / 0.296 / 0.439; +adversary 0.448 / 0.295 / 0.617.

**Verdict — yes the model extracts [mechanics+story], but the adversary is NOT
why.** Both variants keep ~82–84% of content signal while keeping only 49–68% of
sentiment, so the representation *does* preferentially retain story/mechanics over
opinion. **But the cause is VICReg's 1024→18 compression, not the GRL adversary.**
Pure VICReg drops sentiment *harder* (R² 0.44 vs 0.62) and has the *larger*
selectivity gap (+0.335 vs +0.162); adding the adversary actually *increased*
sentiment leakage.

**Why the adversary is inert here:** at `--adversary-weight 1` the entropy term
(~0.4) is ~5% of the total loss (VICReg ≈ 8). The GRL barely steers the encoder.
To give the adversary a real chance, raise `--adversary-weight` to ~10–50 (and/or
`--grl-lambda`), retrain, and re-run `probe_selectivity.py`. The target metric is
the selectivity gap; right now the adversary needs to *beat* the pure-VICReg
baseline of **+0.335**, which it does not.

## Reproduce

```powershell
# Raw-embedding ceiling (219 tags):
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/ceiling_diagnostic.py

# Fair CV probe on a frozen checkpoint:
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/train_tag_probe.py `
  --checkpoint VICReg_review/heads/gui_run/vicreg_review_h5_latest_3.pt `
  --pool flatten --device cuda --amp

# Coarser label sets:
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/tag_build.py --source genres --out-dir VICReg_review/tags_genres
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/ceiling_diagnostic.py --tags-dir VICReg_review/tags_genres

# Selectivity probe (the hypothesis test): content vs subjective vs sentiment.
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/tag_groups.py
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/probe_selectivity.py --device cuda `
  --vic-cache VICReg_review/tags/probe_feat_vicreg_review_h5_latest_3_fv4_sf0.6.npz
```
