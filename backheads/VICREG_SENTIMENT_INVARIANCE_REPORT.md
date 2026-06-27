# VICReg sentiment / identity report

## Final checkpoint

`C:\Users\admin\Documents\studable query latent\VICReg_review\heads\hierarchical64_align_reco\vicreg_review_h5_latest.pt`

## What changed

The latest model is no longer a plain slot-level VICReg encoder. It uses:

- hierarchical latent self-attention and reduction stages
- centroid-level VICReg on a 64-d compact game vector
- full-text description alignment
- recommendation-rate decorrelation

## Headline metrics

| Metric | Result |
|---|---:|
| compact centroid Participation Ratio | 26.67 |
| z-scored compact centroid PR | 26.64 |
| Cyberpunk / AO diagnostic text ranks | 1 / 1 |
| TAP tag micro-F1, flatten pool | 0.6938 |
| content retention vs raw | 0.888 |
| sentiment R² retention vs raw | 0.349 |
| recommendation-rate probe Pearson (CV / holdout) | -0.068 / 0.089 |

## Interpretation

- Content is still readable.
- Sentiment and recommendation-rate linear readouts are strongly suppressed.
- Identity retrieval on the aligned diagnostic texts is rank-1 across Cyberpunk
  and Across the Obelisk, but those long-text variants were included in
  the description-alignment cache, so this should be read as an aligned target
  success rather than pure zero-shot generalization.

## Notes

- This replaces the earlier `hierarchical64_identity` summary.
- The old low-rank collapse case is repaired: the compact centroid is no longer
  stuck near PR ~2.
