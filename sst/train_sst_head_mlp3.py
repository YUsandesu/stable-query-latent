"""Train a single-hidden-layer regression head 1024 -> H -> 1 on SST sentence
embeddings, to sweep width. Default H=128 (sits between the 64-wide MLP1 and
the 618-wide first run, which showed clear capacity saturation).

Early-stops on dev MSE, saves the best checkpoint, then reports MSE / Pearson /
Spearman on test using that best checkpoint. See _sst_train.py for the shared loop.
"""

import argparse
from pathlib import Path

import torch.nn as nn

from _sst_train import train_head

SCRIPT_DIR = Path(__file__).resolve().parent


class Mlp3Head(nn.Module):
    """1024 -> 618 -> 1, GELU + dropout in the hidden layer, sigmoid output.
    Same shape family as Mlp1Head, just much wider (618 vs 64) -- ~10x params."""

    def __init__(self, input_dim=1024, hidden=128, dropout=0.2):
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
    parser.add_argument("--hidden", default=128, type=int)
    parser.add_argument("--checkpoint", default=None,
                        help="Default: <script_dir>/clean/heads/mlp3_1024_<hidden>_1_best.pt")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--max-epochs", default=300, type=int)
    parser.add_argument("--patience", default=30, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    model = Mlp3Head(hidden=args.hidden, dropout=args.dropout)
    print(f"model: {model}")
    checkpoint = args.checkpoint or str(SCRIPT_DIR / f"clean/heads/mlp3_1024_{args.hidden}_1_best.pt")
    train_head(
        model,
        checkpoint_path=checkpoint,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
