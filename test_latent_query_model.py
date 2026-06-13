import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from latent_query_model import LatentQueryFlatRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
SCORE_CLASS_COUNT = 5


def resolve_script_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


class H5LatentQueryDataset(Dataset):
    def __init__(self, h5_path, indices, target_mean=None, target_std=None):
        self.h5_path = Path(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.target_mean = target_mean
        self.target_std = target_std
        self.handle = None

    def _h5(self):
        if self.handle is None:
            self.handle = h5py.File(self.h5_path, "r")
        return self.handle

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        row = int(self.indices[item])
        h5 = self._h5()
        tokens = torch.from_numpy(h5["inputs"][row]).float()
        key_padding_mask = torch.from_numpy(h5["key_padding_mask"][row]).bool()
        target = torch.from_numpy(h5["targets"][row]).float()

        if self.target_mean is not None and self.target_std is not None:
            target = (target - self.target_mean) / self.target_std

        return tokens, key_padding_mask, target

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class TensorLatentQueryDataset(Dataset):
    def __init__(self, inputs, key_padding_masks, targets, indices, target_mean=None, target_std=None):
        indices = np.asarray(indices, dtype=np.int64)
        self.inputs = torch.from_numpy(inputs[indices]).float()
        self.key_padding_masks = torch.from_numpy(key_padding_masks[indices]).bool()
        self.targets = torch.from_numpy(targets[indices]).float()

        if target_mean is not None and target_std is not None:
            self.targets = (self.targets - target_mean) / target_std

    def __len__(self):
        return self.targets.size(0)

    def __getitem__(self, item):
        return self.inputs[item], self.key_padding_masks[item], self.targets[item]

    def close(self):
        pass


def parse_query_sizes(value):
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def make_split(sample_count, test_ratio, seed):
    generator = np.random.default_rng(seed)
    indices = generator.permutation(sample_count)
    test_count = max(1, int(round(sample_count * test_ratio)))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    if len(train_indices) == 0:
        raise ValueError("Train split is empty; reduce --test-ratio.")
    return train_indices, test_indices


def decode_h5_strings(values):
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def make_target_combo_ids(targets):
    targets = np.asarray(targets)
    return np.asarray(["|".join(f"{value:g}" for value in row) for row in targets])


def make_group_split(group_ids, test_ratio, seed):
    group_ids = np.asarray(group_ids)
    unique_groups = np.unique(group_ids)
    generator = np.random.default_rng(seed)
    shuffled_groups = generator.permutation(unique_groups)
    test_group_count = max(1, int(round(len(unique_groups) * test_ratio)))
    test_groups = set(shuffled_groups[:test_group_count].tolist())
    test_mask = np.asarray([group_id in test_groups for group_id in group_ids])
    test_indices = np.flatnonzero(test_mask)
    train_indices = np.flatnonzero(~test_mask)
    if len(train_indices) == 0:
        raise ValueError("Train split is empty; reduce --test-ratio.")
    return train_indices, test_indices, len(unique_groups), len(test_groups)


def targets_to_classes(targets):
    classes = targets.long() - 1
    if classes.min().item() < 0 or classes.max().item() >= SCORE_CLASS_COUNT:
        raise ValueError("Classification targets must contain integer scores from 1 to 5.")
    return classes


def evaluate(model, loader, score_dim, device, pin_memory):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_correct = 0.0
    total_count = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for tokens, key_padding_mask, targets in loader:
            tokens = tokens.to(device, non_blocking=pin_memory)
            key_padding_mask = key_padding_mask.to(device, non_blocking=pin_memory)
            targets = targets.to(device, non_blocking=pin_memory)
            target_classes = targets_to_classes(targets)

            logits = model(tokens, key_padding_mask=key_padding_mask).view(
                targets.size(0), score_dim, SCORE_CLASS_COUNT
            )
            total_loss += criterion(
                logits.reshape(-1, SCORE_CLASS_COUNT),
                target_classes.reshape(-1),
            ).item()

            predicted_scores = logits.argmax(dim=-1).float() + 1.0
            total_mae += torch.abs(predicted_scores - targets).sum().item()
            total_correct += (predicted_scores == targets).sum().item()
            total_count += targets.numel()

    return total_loss / total_count, total_mae / total_count, total_correct / total_count


def train_and_test(
    h5_path,
    epochs,
    batch_size,
    learning_rate,
    min_learning_rate,
    test_ratio,
    seed,
    hidden_dim,
    flat_dim,
    query_sizes,
    num_heads,
    dropout,
    device_name,
    model_out,
    history_txt,
    preload_data,
    split_by,
):
    torch.manual_seed(seed)
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))

    h5_path = resolve_script_path(h5_path)
    if model_out:
        model_out = resolve_script_path(model_out)
    if history_txt:
        history_txt = resolve_script_path(history_txt)

    with h5py.File(h5_path, "r") as h5:
        sample_count = h5["inputs"].shape[0]
        input_dim = h5["inputs"].shape[2]
        score_dim = h5["targets"].shape[1]
        game_ids = decode_h5_strings(h5["benchmark_game_id"][:]) if "benchmark_game_id" in h5 else None
        if preload_data:
            all_inputs = h5["inputs"][:]
            all_key_padding_masks = h5["key_padding_mask"][:]
            all_targets_np = h5["targets"][:]
        else:
            all_targets_np = h5["targets"][:]
    all_targets = torch.from_numpy(all_targets_np).float()

    split_note = "row"
    if split_by == "score_combo":
        split_group_ids = make_target_combo_ids(all_targets_np)
        train_indices, test_indices, group_count, test_group_count = make_group_split(
            split_group_ids, test_ratio, seed
        )
        split_note = f"score_combo groups={group_count} test_groups={test_group_count}"
    elif split_by == "game_id" and game_ids is not None:
        train_indices, test_indices, group_count, test_group_count = make_group_split(
            game_ids, test_ratio, seed
        )
        split_note = f"game_id groups={group_count} test_groups={test_group_count}"
    else:
        train_indices, test_indices = make_split(sample_count, test_ratio, seed)
    if batch_size < 1:
        batch_size = len(train_indices)

    if preload_data:
        train_dataset = TensorLatentQueryDataset(
            all_inputs, all_key_padding_masks, all_targets_np, train_indices
        )
        test_dataset = TensorLatentQueryDataset(
            all_inputs, all_key_padding_masks, all_targets_np, test_indices
        )
    else:
        train_dataset = H5LatentQueryDataset(h5_path, train_indices)
        test_dataset = H5LatentQueryDataset(h5_path, test_indices)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    print(
        f"device={device} samples={sample_count} train={len(train_indices)} "
        f"test={len(test_indices)} batch_size={batch_size} split={split_note} "
        f"preload_data={preload_data}"
    )

    model = LatentQueryFlatRegressor(
        input_dim=input_dim,
        output_dim=score_dim * SCORE_CLASS_COUNT,
        hidden_dim=hidden_dim,
        flat_dim=flat_dim,
        query_sizes=query_sizes,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=min_learning_rate,
    )
    criterion = torch.nn.CrossEntropyLoss()

    history = []
    best_checkpoint = None
    best_metrics = None
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_item_count = 0

            for tokens, key_padding_mask, targets in train_loader:
                tokens = tokens.to(device, non_blocking=pin_memory)
                key_padding_mask = key_padding_mask.to(device, non_blocking=pin_memory)
                targets = targets.to(device, non_blocking=pin_memory)
                target_classes = targets_to_classes(targets)

                optimizer.zero_grad(set_to_none=True)
                logits = model(tokens, key_padding_mask=key_padding_mask).view(
                    targets.size(0), score_dim, SCORE_CLASS_COUNT
                )
                loss = criterion(
                    logits.reshape(-1, SCORE_CLASS_COUNT),
                    target_classes.reshape(-1),
                )
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * targets.numel()
                train_item_count += targets.numel()

            train_ce = train_loss_sum / train_item_count
            test_ce, test_mae, test_accuracy = evaluate(
                model, test_loader, score_dim, device, pin_memory
            )
            current_lr = scheduler.get_last_lr()[0]
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": current_lr,
                    "train_ce": train_ce,
                    "test_ce": test_ce,
                    "test_mae": test_mae,
                    "test_accuracy": test_accuracy,
                }
            )
            if best_metrics is None or test_mae < best_metrics["test_mae"]:
                best_metrics = history[-1]
                best_checkpoint = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            print(
                f"epoch={epoch:03d} "
                f"lr={current_lr:.8f} "
                f"train_ce={train_ce:.6f} "
                f"test_ce={test_ce:.6f} "
                f"test_mae_raw={test_mae:.6f} "
                f"test_acc={test_accuracy:.6f}"
            )
            scheduler.step()
    finally:
        train_dataset.close()
        test_dataset.close()

    if model_out:
        checkpoint = {
            "model_state_dict": best_checkpoint if best_checkpoint is not None else model.state_dict(),
            "input_dim": input_dim,
            "score_dim": score_dim,
            "output_dim": score_dim * SCORE_CLASS_COUNT,
            "score_class_count": SCORE_CLASS_COUNT,
            "hidden_dim": hidden_dim,
            "flat_dim": flat_dim,
            "query_sizes": query_sizes,
            "num_heads": num_heads,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "min_learning_rate": min_learning_rate,
            "seed": seed,
            "test_ratio": test_ratio,
            "split": split_note,
            "split_by": split_by,
            "best_metrics": best_metrics,
            "history": history,
        }
        torch.save(checkpoint, model_out)

    if history_txt and history:
        history_txt.parent.mkdir(parents=True, exist_ok=True)
        with history_txt.open("w", encoding="utf-8") as file:
            file.write("epoch\tlearning_rate\ttrain_ce\ttest_ce\ttest_mae\ttest_accuracy\n")
            for row in history:
                file.write(
                    f"{row['epoch']}\t"
                    f"{row['learning_rate']:.10g}\t"
                    f"{row['train_ce']:.10g}\t"
                    f"{row['test_ce']:.10g}\t"
                    f"{row['test_mae']:.10g}\t"
                    f"{row['test_accuracy']:.10g}\n"
                )

    return best_metrics if best_metrics is not None else (history[-1] if history else None)


def main():
    parser = argparse.ArgumentParser(
        description="Train/test latent_query_model as per-score 5-class classifier from HDF5."
    )
    parser.add_argument("--input-h5", default="benchmark_sentence_latent_query_multi.h5")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Samples per batch. Use 0 to put the whole train split in one batch.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--split-by",
        choices=["score_combo", "game_id", "row"],
        default="score_combo",
        help="Split test data by full target score combo, game_id, or raw rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--flat-dim", type=int, default=128)
    parser.add_argument(
        "--query-sizes",
        type=parse_query_sizes,
        default=(32, 16, 8),
        help="Comma-separated latent query counts, for example 32,16,8.",
    )
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-out", default="latent_query_benchmark_multi_classifier.pt")
    parser.add_argument("--history-txt", default="latent_query_training_history_multi_classifier.txt")
    parser.add_argument(
        "--no-preload-data",
        action="store_true",
        help="Read samples lazily from HDF5 instead of loading the small dataset into RAM first.",
    )
    args = parser.parse_args()

    final_metrics = train_and_test(
        h5_path=args.input_h5,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        test_ratio=args.test_ratio,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        flat_dim=args.flat_dim,
        query_sizes=args.query_sizes,
        num_heads=args.num_heads,
        dropout=args.dropout,
        device_name=args.device,
        model_out=args.model_out,
        history_txt=args.history_txt,
        preload_data=not args.no_preload_data,
        split_by=args.split_by,
    )
    if final_metrics:
        print(
            "best: "
            f"train_ce={final_metrics['train_ce']:.6f}, "
            f"test_ce={final_metrics['test_ce']:.6f}, "
            f"test_mae_raw={final_metrics['test_mae']:.6f}, "
            f"test_acc={final_metrics['test_accuracy']:.6f}"
        )
        print(f"saved model checkpoint: {args.model_out}")
        print(f"saved training history txt: {args.history_txt}")


if __name__ == "__main__":
    main()
