"""Train a deeper regression head with a narrowing bottleneck on SST sentence
embeddings. Default is 1024 -> 64 -> 32 -> 18 -> 1; pass --hidden-dims to try
more aggressive shapes (e.g. --hidden-dims 64 16 4 for 1024->64->16->4->1).

Early-stops on dev MSE, saves the best checkpoint, then reports MSE / Pearson /
Spearman on test using that best checkpoint. See _sst_train.py for the shared loop.
"""

import argparse
from pathlib import Path

import torch.nn as nn

from _sst_train import train_head

SCRIPT_DIR = Path(__file__).resolve().parent


class Mlp4Head(nn.Module):
    """Stack of Linear+GELU+Dropout blocks, narrowing through ``hidden_dims``,
    sigmoid output. e.g. hidden_dims=(64, 32, 18) -> 1024 -> 64 -> 32 -> 18 -> 1."""

    def __init__(self, input_dim=1024, hidden_dims=(64, 32, 18), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 32, 18])
    parser.add_argument("--checkpoint", default=None,
                        help="Default: <script_dir>/clean/heads/mlp4_1024_<dims>_1_best.pt")
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
    model = Mlp4Head(hidden_dims=tuple(args.hidden_dims), dropout=args.dropout)
    print(f"model: {model}")
    dims_tag = "_".join(str(d) for d in args.hidden_dims)
    checkpoint = args.checkpoint or str(SCRIPT_DIR / f"clean/heads/mlp4_1024_{dims_tag}_1_best.pt")
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
