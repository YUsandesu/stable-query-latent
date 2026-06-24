# backheads: Steam recommendation-rate heads

This task trains a supervised head that predicts a game's Steam positive and
negative recommendation rates from game-description/review embeddings.

Pipeline:

1. `VICReg_review/h5/game_review_cleaned_3_sentences.h5`
   supplies per-game Qwen sentence vectors. The cleaned_3 input already includes
   `detailed_description`, `about_the_game`, `short_description`, then reviews.
2. Raw Steam review CSVs supply the label from the `recommend` column:
   `Recommended` -> positive, `Not Recommended` -> negative.
3. `train_recommendation_linear_probe.py` builds or reuses a feature cache,
   trains a ridge linear probe, cross-validates it, and saves a deployable
   checkpoint. `train_recommendation_head.py` is kept as the older MLP-head
   baseline.

Use the project Python interpreter:

```powershell
& 'C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe' backheads/train_recommendation_linear_probe.py
```

Useful options:

```powershell
# Rebuild cached aggregate features from H5
& 'C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe' backheads/train_recommendation_linear_probe.py --overwrite-cache

# Use only long reviews when computing the label, matching the cleaned_3 text filter
& 'C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe' backheads/train_recommendation_linear_probe.py --label-min-length 300

# Predict from a saved checkpoint and existing feature cache
& 'C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe' backheads/predict_recommendation.py --top 20

# Predict directly from one or more text files (local Qwen embedding by default)
& 'C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe' backheads/predict_text_recommendation.py 2077_text.txt AO_text.txt --backend local --device cuda
```

Outputs are written under `backheads/heads/` and ignored by the repo-wide
artifact rules (`*.pt`, `*.npz`, `*.json`, `*.log`).
