"""Train the shallow regression head 1024 -> 64 -> 1 on SST sentence embeddings.

Early-stops on dev MSE, saves the best checkpoint, then reports MSE / Pearson /
Spearman on test using that best checkpoint. See _sst_train.py for the shared loop.
"""

import argparse
from pathlib import Path

import torch.nn as nn

from _sst_train import train_head

SCRIPT_DIR = Path(__file__).resolve().parent


class Mlp1Head(nn.Module):
    """1024 -> 64 -> 1, GELU + dropout in the hidden layer, sigmoid to keep the
    output in [0, 1] (SST labels are continuous in that range)."""

    def __init__(self, input_dim=1024, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(SCRIPT_DIR / "clean/heads/mlp1_1024_64_1_best.pt"))
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--max-epochs", default=300, type=int)
    parser.add_argument("--patience", default=30, type=int,
                        help="Stop if dev MSE does not improve for this many epochs.")
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    model = Mlp1Head(dropout=args.dropout)
    print(f"model: {model}")
    train_head(
        model,
        checkpoint_path=args.checkpoint,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
