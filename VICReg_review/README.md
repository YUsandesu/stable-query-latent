# VICReg Review

Self-supervised game-review encoder:

- Read unified game-review data directly from `game_review_data/embedding_h5.h5`.
- Build two views per game by independently sampling 60 percent of reviews.
- Encode both views with one shared `LatentArrayMLP` (`Latent_Array_MLP` alias):
  input `1024 -> latent_dim 256`, 256 learnable query slots, a single
  cross-attention layer (no residuals, no extra blocks), then a per-latent funnel.
  The current best checkpoint uses `256 -> 128 -> 64`, so the compact downstream
  game centroid is 64-d.
- H5 training now mean-pools latent slots before the loss: `(B, 256, D) -> (B, D)`.
  Invariance is computed on these compact game centroids. Variance/covariance are
  computed on an expander MLP projection (`D -> 512`) so the regularizer acts on
  inter-game separation instead of being satisfied by within-game slot spread.
- Optional compact variance/covariance auxiliary terms keep the downstream
  centroid itself high-rank.
- Apply a frozen SST MLP4-A sentiment head through GRL so the latent codes become
  sentiment-confusing (driven toward output 0.5 = maximum entropy). Because the
  head needs 1024-d inputs, the adversary holds a learnable up-projection probe
  (`D -> 256 -> 1024`) placed *after* the GRL, so the encoder is always the
  adversarial party while the probe tries to recover sentiment confidence.

Default run:

```powershell
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/train_vicreg_review_h5.py --input-h5 game_review_data/embedding_h5.h5 --device cuda --amp
```

Useful smoke test:

```powershell
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/train_vicreg_review_h5.py --input-h5 game_review_data/embedding_h5.h5 --device cpu --epochs 1 --steps-per-epoch 1 --batch-size 1 --limit-games 1 --no-save
```

Outputs are written under `VICReg_review/heads/` by default. Checkpoints and JSON
manifests are ignored by the project `.gitignore`.

Build the H5 corpus, then train:

```powershell
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe game_review_data/build.py --backend cloud
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/train_vicreg_review_h5.py --device cuda --amp `
  --epochs 30 --steps-per-epoch 4 --batch-size 128 `
  --vicreg-scope game --output-dim 64 --reduce-hidden 128 `
  --expander-dim 512 --expander-hidden 256,512 `
  --compact-variance-weight 25 --compact-covariance-weight 25
```

`game_review_data/build.py` writes TAP labels into the H5:
`tap_names`, `tap_labels`, `tap_raw_counts`, `appids`, and `game_titles`.
The only tag mapping source is `VICReg_review/tags/tap_mapping.json`, where each
fine Steam tag maps to one coarse TAP class or `"del"`.
`build_review_h5.py` is now only a legacy converter for old embedded JSON
corpora.

Use `--cache-mode full` only when RAM can hold the next prepared epoch. The
default `queue` mode overlaps H5 loading with GPU training using a bounded
prefetch queue.

## Tag validation probe (diagnostic only)

A separate, **validation-only** head checks whether the frozen encoder code can
predict a game's Steam tags. It never touches the VICReg loss — the encoder is
loaded frozen and the probe has its own optimizer and self-stops when learning
plateaus. Rising tag mAP across VICReg checkpoints = a more robust representation.
The probe flattens the `(256, 18)` code (4608 dims) before its MLP.

```powershell
# Probe a checkpoint: review text -> frozen encoder -> TAP labels.
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/train_tag_probe.py `
  --device cuda --amp --checkpoint VICReg_review/heads/vicreg_review_h5_best.pt
```

`train_tag_probe.py` (redesigned) caches one averaged feature per game
(`--feature-views`), pools it (`--pool flatten|stats|mean`), and runs **k-fold
cross-validation** with a **fairness rule** (a tag is scored only where it has
`--min-train-pos` train positives) per-tag logistic regression. It reports
micro-F1 mean±std, a frequency-floor breakdown, and (with `--export-head PATH`)
saves a portable linear probe for `validation.py`.

See `TAG_PROBE_RESULTS.md` for the full analysis: the raw-embedding F1 ceiling,
the old 18-d bottleneck, and the current 64-d game-centroid results.

## Current best checkpoint

`heads/hierarchical64_align_reco/vicreg_review_h5_latest.pt` is the current best
checkpoint. It uses a hierarchical latent-array encoder with self-attention and
reduction stages, a 64-d compact game centroid, centroid-level VICReg, full-text
description alignment, and a recommendation-rate decorrelation term.

Headline diagnostics:

| Metric | Result |
|---|---:|
| compact centroid Participation Ratio | 26.67 |
| z-scored compact centroid PR | 26.64 |
| Cyberpunk 2077 neutral / positive / negative / noname ranks | 1 / 1 / 1 / TBD |
| Across the Obelisk neutral / positive / negative / noname ranks | 1 / 1 / 1 / TBD |
| TAP tag micro-F1, flatten pool | 0.6938 |
| content retention vs raw | 0.888 |
| sentiment R² retention vs raw | 0.349 |
| recommendation-rate linear probe Pearson (CV / holdout) | -0.068 / 0.089 |

Updated artifacts:

- `heads/hierarchical64_align_reco/tag_probe_linear_flatten.pt`
- `heads/hierarchical64_align_reco/tag_probe_report_flatten.json`
- `heads/hierarchical64_align_reco/identity_diagnostic_report.json`
- `heads/hierarchical64_align_reco/selectivity_report.json`
- `backheads/heads/recommendation_vicreg_features_hierarchical64_align_reco.npz`
- `backheads/heads/recommendation_vicreg_linear_probe_hierarchical64_align_reco.pt`
- `backheads/heads/recommendation_vicreg_linear_probe_hierarchical64_align_reco_report.json`

## Dual-probe validation during training

`train_vicreg_review_h5.py` runs a **tag + PXI dual probe** on the live encoder
every `--probe-every` epochs (**default 1** — on; set 0 for smoke tests) and on the
last epoch, logging a validation curve to `heads/dual_probe_history.tsv`. Per probe it reports, all from the frozen code:
`tag_content_f1`, `tag_subjective_f1`, `tag_selectivity` (= content − subjective),
`code_sentiment_r2` (sentiment suppression), and `pxi_func_f1` / `pxi_psych_f1`
(functional vs psychological, on the 21 PXI-overlap games). The probe uses the
`stats` pool for speed, restores the encoder's train mode, and never aborts
training on failure. Standalone: `python dual_probe.py --checkpoint ...`.

```powershell
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/train_vicreg_review_h5.py `
  --device cuda --amp --probe-every 5
```

## Hypothesis test: keeps mechanics+story, drops sentiment

The real claim is that the encoder + sentiment adversary retains content and
filters opinion. `probe_selectivity.py` measures content-tag F1, subjective-tag
F1, and SST-sentiment R² on the VICReg code vs the raw embedding (ceiling) and
reports retention = vicreg/raw. TAP labels are already coarse/non-subjective, so
they are treated as content labels. `ceiling_diagnostic.py` is the raw-embedding
upper bound.

```powershell
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/probe_selectivity.py --device cuda `
  --vic-cache VICReg_review/tags/probe_feat_<checkpoint>_fv4_sf0.6.npz
```

## Validation UI (`validation.py`)

Aligned with the current probe. Three steps: export the deployable linear probe,
build the in-domain game pool, then launch the UI (it auto-loads both):

```powershell
# 1. export the linear probe from the best encoder.
#    Use --pool stats for deployment: flatten has ~3.7k near-zero-variance dims that
#    make StandardScaler explode on out-of-distribution text (saturated prob=1 tags).
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/train_tag_probe.py `
  --checkpoint VICReg_review/heads/sweep_adv/vicreg_adv10_best.pt --pool stats `
  --device cpu --export-head VICReg_review/heads/tag_probe_linear.pt

# 2. launch
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe validation.py
```

Pipeline: text → local Qwen embedding → frozen encoder code → pool → L2 normalize →
per-TAP logistic probe → TAP probabilities. Game candidates come from the same
H5 `tap_raw_counts` matrix used to train/evaluate the probe, so matching cannot
drift from the label mapping.
