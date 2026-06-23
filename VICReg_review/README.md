# VICReg Review

Self-supervised game-review encoder:

- Pick one game JSON from `game_review_data/game_review_cleaned_3_sentences`.
- Build two views by independently sampling 60 percent of reviews.
- Encode both views with one shared `LatentArrayMLP` (`Latent_Array_MLP` alias).
- Train with VICReg consistency on the final latent array.
- Apply a frozen SST MLP4-A sentiment head through GRL so the 16 final 1024-d
  latent vectors become sentiment-confusing.

Default run:

```powershell
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/train_vicreg_review.py --device cuda --amp
```

Useful smoke test:

```powershell
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/train_vicreg_review.py --device cpu --epochs 1 --steps-per-epoch 1 --limit-games 1 --max-sentences 8 --no-save
```

Outputs are written under `VICReg_review/heads/` by default. Checkpoints and JSON
manifests are ignored by the project `.gitignore`.

HDF5 path for faster training:

```powershell
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/build_review_h5.py --workers 2 --shards 8
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/train_vicreg_review_h5.py --device cuda --amp --epochs 100 --batch-size 16
```

Use `--cache-mode full` only when RAM can hold the next prepared epoch. The
default `queue` mode overlaps H5 loading with GPU training using a bounded
prefetch queue.
