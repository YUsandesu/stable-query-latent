# VICReg PXI Result

Generated: 2026-06-24 20:36:23

## Scope

- Input: `VICReg_review/h5/game_review_cleaned_3_sentences.h5` game vectors. This H5 comes from cleaned_3, where Steam metadata/description fields are prepended before reviews.
- Intersection: 21 PXIbenchmark games with VICReg H5 rows.
- Path: cleaned_3 H5 game view -> frozen VICReg -> cached VICReg code -> PXI mean-regression head.
- Feature cache: `VICReg_review/tags/pxi_feat_h5_cleaned3_vicreg_adv10_best_fv4_sf0.6.npz`.
- Raw direct baseline cache: `VICReg_review/tags/pxi_feat_h5_cleaned3_raw_direct.npz`.
- Description-only baseline cache: `VICReg_review/tags/pxi_feat_game_descriptions_raw_direct.npz`.
- Exported head: `VICReg_review/heads/pxi_probe_linear.pt`.
- Best LOO config: pool=mean, normalizer=standard, pca=2, alpha=30.0.

## Raw Embedding Baseline

This baseline bypasses VICReg. It directly pools every cleaned_3 sentence embedding for each game (the prepended Steam description metadata plus all reviews), then fits a linear/Ridge PXI head with leave-one-out testing. It is the raw-Qwen reference for how much PXI signal is available before the VICReg bottleneck.

| feature set | normalizer | pca | alpha | LOO MAE | RMSE | Pearson | R^2 | clipped raw values |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw_mean | standard | 2 | 1000 | 0.697 | 0.881 | 0.297 | 0.064 | 0/210 |

## Description-Only Raw Baseline

This baseline embeds only `VICReg_review/tags/game_descriptions/{appid}.txt` for each PXI overlap game, pools the resulting Qwen sentence embeddings, and fits a linear/Ridge PXI head with leave-one-out testing. It answers how far the public game description text alone can go without reviews or VICReg.

| feature set | normalizer | pca | alpha | LOO MAE | RMSE | Pearson | R^2 | clipped raw values |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw_mean | l2 | 2 | 0.01 | 0.689 | 0.871 | 0.314 | 0.087 | 0/210 |

## MLP Probe Experiment

This probe puts a small MLP after the VICReg code, matching the idea that downstream tasks often need a learned adapter. The table reports strict leave-one-out performance, so improvements here count only if they beat the linear VICReg probe on held-out games.

| pool | hidden dims | dropout | weight decay | lr | LOO MAE | RMSE | Pearson | R^2 | clipped raw values |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stats | 8,4 | 0.1 | 1 | 0.001 | 0.709 | 0.908 | 0.290 | 0.007 | 0/210 |

## Training Fit Metrics

These numbers are computed after fitting the exported PXI head on all 21 games. They describe how well the head fits the available cleaned_3 input (Steam description metadata plus review text). Lower MAE is better.

| subset | N values | MAE | RMSE | Pearson | R^2 | clipped raw values |
|---|---:|---:|---:|---:|---:|---:|
| all dimensions | 210 | 0.614 | 0.782 | 0.514 | 0.262 | 0/210 |
| functional | 105 | 0.567 | 0.767 | 0.592 | 0.350 | 0/210 |
| psychological | 105 | 0.661 | 0.797 | 0.399 | 0.153 | 0/210 |

## Leave-One-Out Test Metrics

These numbers are the stricter test estimate: each game is predicted by a head trained on the other 20 games, using the same selected configuration.

| subset | N values | MAE | RMSE | Pearson | R^2 | clipped raw values |
|---|---:|---:|---:|---:|---:|---:|
| all dimensions | 210 | 0.700 | 0.897 | 0.270 | 0.031 | 0/210 |
| functional | 105 | 0.647 | 0.879 | 0.411 | 0.146 | 0/210 |
| psychological | 105 | 0.754 | 0.914 | -0.003 | -0.112 | 0/210 |

## Per-Game LOO Predictions

| appid | PXI game | Steam game | match | PXI samples | MAE | functional MAE | psychological MAE |
|---:|---|---|---|---:|---:|---:|---:|
| 1145360 | Hades | Hades | exact | 13 | 0.151 | 0.141 | 0.160 |
| 1284210 | Guild Wars 2 | Guild Wars 2 | exact | 13 | 0.322 | 0.327 | 0.317 |
| 1091500 | Cyberpunk 2077 | Cyberpunk 2077 | exact | 8 | 0.347 | 0.351 | 0.344 |
| 1817070 | Spider-Man | Marvel’s Spider-Man Remastered | vetted_variant | 2 | 0.484 | 0.516 | 0.452 |
| 1113560 | Nier Replicant | NieR Replicant™ ver.1.22474487139... | vetted_variant | 1 | 0.491 | 0.442 | 0.540 |
| 1449560 | Metro Exodus | Metro Exodus | exact | 1 | 0.534 | 0.643 | 0.426 |
| 1593500 | God of War | God of War | exact | 3 | 0.641 | 0.562 | 0.720 |
| 1262540 | Need for Speed | Need for Speed™ | exact | 4 | 0.647 | 0.586 | 0.708 |
| 1151640 | Horizon zero dawn | Horizon Zero Dawn™ Complete Edition | vetted_variant | 3 | 0.651 | 0.728 | 0.573 |
| 1172470 | Apex Legends | Apex Legends™ | exact | 3 | 0.678 | 0.524 | 0.832 |
| 1343400 | RuneScape | RuneScape ® | exact | 4 | 0.705 | 0.779 | 0.631 |
| 1237950 | Star Wars Battlefront | STAR WARS™ Battlefront™ II | vetted_variant | 1 | 0.729 | 0.554 | 0.904 |
| 1325200 | Nioh 2 | Nioh 2 – The Complete Edition | vetted_variant | 1 | 0.750 | 0.875 | 0.625 |
| 1659420 | Uncharted | UNCHARTED™: Legacy of Thieves Collection | vetted_variant | 1 | 0.804 | 0.995 | 0.614 |
| 1649240 | Returnal | Returnal™ | exact | 1 | 0.866 | 0.694 | 1.039 |
| 1237970 | Titanfall 2 | Titanfall® 2 | exact | 1 | 0.883 | 0.661 | 1.104 |
| 1458100 | Cozy Grove | Cozy Grove | exact | 1 | 0.962 | 1.064 | 0.861 |
| 1407200 | World of Tanks | World of Tanks | exact | 5 | 0.976 | 0.613 | 1.338 |
| 1293830 | Forza Horizon 4 | Forza Horizon 4 | exact | 1 | 1.011 | 0.989 | 1.034 |
| 1549970 | Aliens Fireteam Elite | Aliens: Fireteam Elite | exact | 1 | 1.036 | 0.524 | 1.549 |
| 1124300 | Humankind | HUMANKIND™ | exact | 1 | 1.036 | 1.014 | 1.059 |

## Per-Dimension LOO Metrics

| dimension | group | MAE | RMSE | Pearson | R^2 | actual mean | predicted mean | raw min | raw max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| psychological_meaning | psychological | 0.770 | 0.913 | -0.210 | -0.178 | 1.749 | 1.739 | 1.220 | 2.266 |
| psychological_mastery | psychological | 0.601 | 0.707 | -0.558 | -0.238 | 1.913 | 1.904 | 1.658 | 2.229 |
| psychological_curiosity | psychological | 0.984 | 1.089 | -0.038 | -0.085 | 1.920 | 1.923 | 1.459 | 2.665 |
| psychological_autonomy | psychological | 0.830 | 1.091 | -0.423 | -0.314 | 1.944 | 1.927 | 0.971 | 2.238 |
| psychological_immersion | psychological | 0.584 | 0.683 | 0.447 | 0.195 | 1.656 | 1.675 | 0.846 | 2.366 |
| functional_progress_feedback | functional | 0.652 | 0.819 | -0.279 | -0.169 | 1.775 | 1.794 | 1.516 | 2.319 |
| functional_ease_of_control | functional | 0.452 | 0.610 | -0.280 | -0.151 | 1.881 | 1.886 | 1.590 | 2.094 |
| functional_audiovisual_appeal | functional | 0.478 | 0.627 | -0.086 | -0.162 | 2.546 | 2.549 | 2.104 | 2.900 |
| functional_goals_and_rules | functional | 0.593 | 0.722 | 0.014 | -0.077 | 2.272 | 2.274 | 1.933 | 2.657 |
| functional_challenge | functional | 1.058 | 1.382 | -0.189 | -0.201 | 1.094 | 1.122 | 0.426 | 1.959 |

## Top Description-Only Baseline Configs

| rank | feature set | normalizer | pca | alpha | objective | raw MAE | raw Pearson | clip fraction | mean excess |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | raw_mean | l2 | 2 | 0.01 | 0.689 | 0.689 | 0.314 | 0.000 | 0.000 |
| 2 | raw_mean | l2 | 2 | 0.03 | 0.689 | 0.689 | 0.313 | 0.000 | 0.000 |
| 3 | raw_mean | l2 | 2 | 0.1 | 0.691 | 0.691 | 0.311 | 0.000 | 0.000 |
| 4 | raw_stats | l2 | 2 | 0.01 | 0.691 | 0.691 | 0.312 | 0.000 | 0.000 |
| 5 | raw_stats | l2 | 2 | 0.03 | 0.692 | 0.692 | 0.310 | 0.000 | 0.000 |
| 6 | raw_mean | standard | 2 | 0.01 | 0.692 | 0.692 | 0.307 | 0.000 | 0.000 |
| 7 | raw_mean | standard | 2 | 1 | 0.692 | 0.692 | 0.307 | 0.000 | 0.000 |
| 8 | raw_mean | standard | 2 | 3 | 0.692 | 0.692 | 0.307 | 0.000 | 0.000 |
| 9 | raw_mean | standard | 2 | 0.03 | 0.692 | 0.692 | 0.307 | 0.000 | 0.000 |
| 10 | raw_mean | standard | 2 | 30 | 0.692 | 0.692 | 0.307 | 0.000 | 0.000 |

## Top MLP Probe Configs

| rank | pool | hidden dims | dropout | weight decay | lr | objective | raw MAE | raw Pearson | clip fraction | mean excess |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | stats | 8,4 | 0.1 | 1 | 0.001 | 0.709 | 0.709 | 0.290 | 0.000 | 0.000 |

## Top Raw Baseline Configs

| rank | feature set | normalizer | pca | alpha | objective | raw MAE | raw Pearson | clip fraction | mean excess |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | raw_mean | standard | 2 | 1000 | 0.697 | 0.697 | 0.297 | 0.000 | 0.000 |
| 2 | raw_mean | standard | 2 | 300 | 0.698 | 0.698 | 0.293 | 0.000 | 0.000 |
| 3 | raw_mean | l2 | 2 | 0.1 | 0.698 | 0.698 | 0.294 | 0.000 | 0.000 |
| 4 | raw_mean | standard | 2 | 100 | 0.699 | 0.699 | 0.292 | 0.000 | 0.000 |
| 5 | raw_mean | l2 | 2 | 0.3 | 0.699 | 0.699 | 0.295 | 0.000 | 0.000 |
| 6 | raw_mean | standard | 2 | 30 | 0.699 | 0.699 | 0.291 | 0.000 | 0.000 |
| 7 | raw_mean | l2 | 2 | 0.03 | 0.699 | 0.699 | 0.287 | 0.000 | 0.000 |
| 8 | raw_mean | standard | 2 | 10 | 0.699 | 0.699 | 0.291 | 0.000 | 0.000 |
| 9 | raw_mean | standard | 2 | 3 | 0.699 | 0.699 | 0.291 | 0.000 | 0.000 |
| 10 | raw_mean | standard | 2 | 1 | 0.699 | 0.699 | 0.291 | 0.000 | 0.000 |

## Top Grid Configs

| rank | pool | normalizer | pca | alpha | objective | raw MAE | raw Pearson | clip fraction | mean excess |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mean | standard | 2 | 30 | 0.700 | 0.700 | 0.270 | 0.000 | 0.000 |
| 2 | mean | standard | 2 | 100 | 0.701 | 0.701 | 0.277 | 0.000 | 0.000 |
| 3 | mean | standard | 2 | 300 | 0.702 | 0.702 | 0.284 | 0.000 | 0.000 |
| 4 | stats | l2 | 0 | 10 | 0.702 | 0.702 | 0.287 | 0.000 | 0.000 |
| 5 | flatten | l2 | 0 | 100 | 0.702 | 0.702 | 0.287 | 0.000 | 0.000 |
| 6 | flatten | l2 | 2 | 100 | 0.702 | 0.702 | 0.287 | 0.000 | 0.000 |
| 7 | flatten | l2 | 4 | 100 | 0.702 | 0.702 | 0.287 | 0.000 | 0.000 |
| 8 | stats | l2 | 6 | 30 | 0.702 | 0.702 | 0.287 | 0.000 | 0.000 |
| 9 | stats | l2 | 8 | 30 | 0.702 | 0.702 | 0.287 | 0.000 | 0.000 |
| 10 | stats | l2 | 12 | 30 | 0.702 | 0.702 | 0.287 | 0.000 | 0.000 |

## Caveats

- N is only 21 games, so leave-one-out estimates have high variance.
- The final exported head is fit on all 21 games after selecting the configuration by LOO.
- This head is calibrated for VICReg features built from the cleaned_3 H5 input distribution.
