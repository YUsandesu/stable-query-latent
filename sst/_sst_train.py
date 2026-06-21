"""Shared training utilities for the SST regression heads (used by
train_sst_head_mlp1.py and train_sst_head_mlp2.py).

Reads ``clean/sentence_embeddings/default_{train,dev,test}.json`` (each a list
of {sentence, label, embedding}). Trains a regression head on the train split,
monitors MSE on the dev split for early stopping, saves the best checkpoint, then
reports MSE / Pearson / Spearman on the test split using that best checkpoint.

Early-stopping monitor: **dev**, not test. Using test for early stopping leaks
test information into training and inflates the final number — dev is the right
held-out signal during training, test is reserved for the final report.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DATA_DIR = Path(__file__).resolve().parent / "clean" / "sentence_embeddings"


def load_split(split):
    path = DATA_DIR / f"default_{split}.json"
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    X = np.asarray([record["embedding"] for record in records], dtype=np.float32)
    y = np.asarray([record["label"] for record in records], dtype=np.float32)
    return X, y


def l2_normalize(X):
    """Per-row L2 normalize (Qwen embeddings ship unnormalized; the head trains
    much more stably on unit vectors)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return X / norms


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float((a * b).sum() / denom)


def spearman(a, b):
    return pearson(a.argsort().argsort().astype(np.float64),
                   b.argsort().argsort().astype(np.float64))


def _to_tensor(array, device):
    return torch.from_numpy(array).to(device)


def train_head(
    model,
    checkpoint_path,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=64,
    max_epochs=300,
    patience=30,
    min_delta=1e-5,
    device=None,
    seed=0,
):
    """Train ``model`` on SST default_train, early-stop on default_dev MSE, save
    the best state to ``checkpoint_path``, and report the test result. Returns a
    dict with the best dev metrics and final test metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr = load_split("train")
    X_dv, y_dv = load_split("dev")
    X_te, y_te = load_split("test")
    X_tr = l2_normalize(X_tr); X_dv = l2_normalize(X_dv); X_te = l2_normalize(X_te)
    print(f"data: train={len(X_tr)} dev={len(X_dv)} test={len(X_te)} | dim={X_tr.shape[1]}")

    X_tr_t = _to_tensor(X_tr, device); y_tr_t = _to_tensor(y_tr, device)
    X_dv_t = _to_tensor(X_dv, device); y_dv_t = _to_tensor(y_dv, device)
    X_te_t = _to_tensor(X_te, device); y_te_t = _to_tensor(y_te, device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_dev_mse = float("inf")
    best_epoch = -1
    no_improve = 0

    print(f"training on {device} | lr={lr} wd={weight_decay} batch={batch_size} patience={patience}")
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=device)
        running = 0.0
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            xb = X_tr_t[idx]; yb = y_tr_t[idx]
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        train_mse = running / len(X_tr_t)

        model.eval()
        with torch.no_grad():
            dev_pred = model(X_dv_t).squeeze(-1)
            dev_mse = loss_fn(dev_pred, y_dv_t).item()

        improved = dev_mse < best_dev_mse - min_delta
        if improved:
            best_dev_mse = dev_mse
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "dev_mse": dev_mse,
                    "model_repr": repr(model),
                },
                checkpoint_path,
            )
            marker = " * (saved)"
        else:
            no_improve += 1
            marker = ""
        print(f"epoch {epoch:3d}: train_mse={train_mse:.4f} dev_mse={dev_mse:.4f} "
              f"no_improve={no_improve}/{patience}{marker}")
        if no_improve >= patience:
            print(f"early stop: dev MSE did not improve for {patience} epochs.")
            break

    # Reload the best checkpoint and report test metrics.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        te_pred = model(X_te_t).squeeze(-1)
        test_mse = loss_fn(te_pred, y_te_t).item()
    te_np = te_pred.cpu().numpy(); y_te_np = y_te_t.cpu().numpy()
    test_pearson = pearson(te_np, y_te_np)
    test_spearman = spearman(te_np, y_te_np)

    print(f"\nBEST  dev_mse={best_dev_mse:.4f} @ epoch {best_epoch}")
    print(f"TEST  mse={test_mse:.4f}  pearson={test_pearson:.4f}  spearman={test_spearman:.4f}")
    print(f"checkpoint: {checkpoint_path}")
    return {
        "best_dev_mse": best_dev_mse,
        "best_epoch": best_epoch,
        "test_mse": test_mse,
        "test_pearson": test_pearson,
        "test_spearman": test_spearman,
        "checkpoint": str(checkpoint_path),
    }
