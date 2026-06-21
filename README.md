# studable-query-latent

Three coupled experiments around sentence-level embeddings:

| Subproject | What it does |
|---|---|
| [`PXIbench_test/`](PXIbench_test/) | Train a **latent-query** model to predict 10 × 5-class PXI scores from pseudo-text sentence vectors. |
| [`sst/`](sst/) | Train tiny MLP heads on **SST** sentiment regression from frozen sentence embeddings. |
| [`game_review_data/`](game_review_data/) | End-to-end pipeline for Steam game-review text → sentence-level vectors. |

The model definition lives at the repo root: [`latent_query_model.py`](latent_query_model.py). Each subdirectory has its own README with task-specific details.

---

## Setup

```bash
# 1. Clone
git clone <repo-url> studable-query-latent
cd studable-query-latent

# 2. Install (use the cuda-enabled env you prefer; project is built/tested
#    against C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe on Windows)
pip install -r requirements.txt
```

### Configure the cloud embedding endpoint

The cloud-embedding code reads its endpoint URL and HuggingFace token from
`tokenAPI.txt` at the repo root. This file is **gitignored** so your real
credentials never get committed.

```bash
# 3. Create your tokenAPI.txt from the tracked template
cp tokenAPI.template.txt tokenAPI.txt
```

Then **open `tokenAPI.txt` and replace the placeholder values**:

```ini
# before (placeholder)
url=https://<your-endpoint-id>.<region>.aws.endpoints.huggingface.cloud
token=hf_<your-token-here>

# after (your real values, e.g.)
url=https://abc1234.us-east-1.aws.endpoints.huggingface.cloud
token=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Notes:
- One `KEY=VALUE` per line, `#` for comments, blank lines OK.
- Both `url=` and `token=` are required; the parser
  ([`game_review_data/cloud_embedding.load_credentials`](game_review_data/cloud_embedding.py))
  raises a clear error if either is missing.
- The placeholder values (the `<...>` parts) are **not** valid endpoints —
  scripts will fail with a 404 / bad-request if you leave them in.
- You can also embed locally without an endpoint:
  `python sst/embed_sst.py --backend local --device cuda`.

If you ever need to share the template again (e.g. you accidentally deleted it),
it lives at [`tokenAPI.template.txt`](tokenAPI.template.txt) — the only
version-controlled credential-shaped file in the repo.

---

## Quick starts

```bash
# Full PXI pipeline (download → pseudo-text → embed → h5) and train
python PXIbench_test/Build_data.py
python PXIbench_test/test_latent_query_model.py

# SST sentiment regression head (best variant per sst/heads/RESULTS.md)
python sst/train_sst_head_mlp4.py --hidden-dims 64 16 4

# Game-review end-to-end build
python game_review_data/build_gamedata.py --workdir <workdir> --backend cloud
```

