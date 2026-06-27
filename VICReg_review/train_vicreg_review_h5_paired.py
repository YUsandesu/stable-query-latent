"""Train GRL and no-GRL review encoders on the exact same H5 batches.

This is a paired variant of train_vicreg_review_h5.py for the final sweep.  For
each epoch/batch it samples review views once, then trains two independent arms:

* grl: adversary_weight > 0, with the usual GRL schedule and recommendation loss.
* nogrl: adversary_weight = 0 and recommendation loss disabled.

The arms keep separate checkpoints, histories, optimizers, and manifests.  The
sampled input views, description samples, recommendation labels, and dropout RNG
sequence are shared per batch so differences are attributable to the GRL arm.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VICReg_review.train_vicreg_review_h5 import (  # noqa: E402
    DEFAULT_DESCRIPTION_CACHE,
    DEFAULT_H5,
    DEFAULT_HEADS_DIR,
    DEFAULT_SST_CHECKPOINT,
    build_training_components,
    capture_rng_state,
    decode_name,
    game_sentence_counts,
    grl_lambda_at,
    iter_epoch,
    load_description_bank,
    load_recommendation_targets,
    make_epoch_indices,
    prepare_epoch_batches,
    parse_int_list,
    parse_string_list,
    resolve_train_game_indices,
    restore_rng_state,
    run_dual_probe,
    run_training_batch,
    should_run_probe,
    write_history,
    write_manifest,
    atomic_torch_save,
)


def arm_namespace(args, arm: str, adversary_weight: float):
    out = copy.copy(args)
    prefix = "grl" if arm == "grl" else "nogrl"
    out.arm = arm
    out.adversary_weight = float(adversary_weight)
    out.recommendation_decorr_weight = float(
        getattr(args, f"{prefix}_recommendation_decorr_weight", getattr(args, "recommendation_decorr_weight", 0.0))
    )
    if out.adversary_weight <= 0:
        out.grl_lambda = 0.0
    out.checkpoint_out = getattr(args, f"{prefix}_checkpoint_out")
    out.best_checkpoint_out = getattr(args, f"{prefix}_best_checkpoint_out")
    out.history_tsv = getattr(args, f"{prefix}_history_tsv")
    out.manifest_json = getattr(args, f"{prefix}_manifest_json")
    out.probe_history_tsv = getattr(args, f"{prefix}_probe_history_tsv")
    out.resume_checkpoint = None
    return out


def init_arm(args, input_dim: int, device, amp_enabled: bool):
    model, expander, adversary, optimizer = build_training_components(args, input_dim, device)
    return {
        "args": args,
        "model": model,
        "expander": expander,
        "adversary": adversary,
        "optimizer": optimizer,
        "scaler": torch.amp.GradScaler("cuda", enabled=amp_enabled),
        "history_rows": [],
        "probe_rows": [],
        "best_loss": float("inf"),
        "global_step": 0,
        "last_metrics": None,
    }


def checkpoint_payload(state, epoch: int, averaged: dict, args, input_dim: int):
    model = state["model"]
    expander = state["expander"]
    adversary = state["adversary"]
    optimizer = state["optimizer"]
    return {
        "model_state_dict": model.state_dict(),
        "expander_state_dict": expander.state_dict() if expander is not None else None,
        "adversary_state_dict": adversary.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": state["global_step"],
        "resumed_from": None,
        "args": vars(args),
        "metrics": averaged,
        "model_class": model.__class__.__name__,
        "vicreg_scope": args.vicreg_scope,
        "num_latents": args.num_latents,
        "latent_dim": args.latent_dim,
        "output_dim": args.output_dim,
        "expander_dim": args.expander_dim if expander is not None else None,
        "expander_hidden": tuple(args.expander_hidden) if expander is not None else None,
        "input_dim": int(input_dim),
        "input_h5": str(Path(args.input_h5).resolve()),
        "sst_checkpoint": str(Path(args.sst_checkpoint).resolve()),
        "paired_training": True,
        "arm": args.arm,
    }


def prepare_common_state(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_dtype = np.dtype(args.cache_dtype)

    with h5py.File(args.input_h5, "r") as h5:
        num_games = int(h5["game_names"].shape[0])
        input_dim = int(h5.attrs["input_dim"])
        total_sentences = int(h5["vectors"].shape[0])
        h5_appids = [decode_name(x) for x in h5["appids"][:]] if "appids" in h5 else [
            decode_name(x).split("_", 1)[0] for x in h5["game_names"][:]
        ]
        train_game_indices = resolve_train_game_indices(args, h5)
        effective_num_games = len(train_game_indices) if train_game_indices is not None else num_games
        if args.max_batch_sentences > 0:
            counts = game_sentence_counts(h5)
            default_steps = len(
                make_epoch_indices(
                    num_games,
                    args.seed + 1,
                    args.batch_size,
                    0,
                    game_order=args.game_order,
                    counts=counts,
                    max_batch_sentences=args.max_batch_sentences,
                    game_indices=train_game_indices,
                )
            )
        else:
            default_steps = math.ceil(effective_num_games / args.batch_size)
    if args.steps_per_epoch <= 0:
        args.steps_per_epoch = default_steps

    description_bank = None
    if args.description_align_weight > 0 or args.description_mse_weight > 0:
        description_bank = load_description_bank(args, h5_appids)
        if args.train_game_indices is not None:
            keep = set(int(i) for i in args.train_game_indices)
            description_bank = [items if index in keep else [] for index, items in enumerate(description_bank)]

    recommendation_weight = max(
        float(getattr(args, "recommendation_decorr_weight", 0.0)),
        float(getattr(args, "grl_recommendation_decorr_weight", 0.0)),
        float(getattr(args, "nogrl_recommendation_decorr_weight", 0.0)),
    )
    recommendation_targets = None
    if recommendation_weight > 0:
        recommendation_targets = load_recommendation_targets(args, num_games)
        if args.train_game_indices is not None:
            mask = np.ones_like(recommendation_targets, dtype=bool)
            mask[np.asarray(args.train_game_indices, dtype=np.int64)] = False
            recommendation_targets[mask] = np.nan

    print(
        f"paired device={device} games={num_games} sentences={total_sentences} "
        f"train_games={effective_num_games} batch_size={args.batch_size} "
        f"steps_per_epoch={args.steps_per_epoch} sample_fraction={args.sample_fraction} "
        f"max_batch_sentences={args.max_batch_sentences} "
        f"max_view_sentences={args.max_view_sentences}",
        flush=True,
    )
    return device, cache_dtype, input_dim, description_bank, recommendation_targets


def train(args) -> None:
    device, cache_dtype, input_dim, description_bank, recommendation_targets = prepare_common_state(args)
    amp_enabled = args.amp and device.type == "cuda"
    pin_transfer = args.pin_cache and device.type == "cuda"

    # Reset the seed before each build so both arms start from identical weights.
    torch.manual_seed(args.seed)
    grl_args = arm_namespace(args, "grl", args.grl_adversary_weight)
    grl_state = init_arm(grl_args, input_dim, device, amp_enabled)
    torch.manual_seed(args.seed)
    nogrl_args = arm_namespace(args, "nogrl", args.nogrl_adversary_weight)
    nogrl_state = init_arm(nogrl_args, input_dim, device, amp_enabled)
    arms = [grl_state, nogrl_state]

    for state in arms:
        expander = state["expander"]
        print(
            f"arm={state['args'].arm} adversary_weight={state['args'].adversary_weight} "
            f"model={state['model'].__class__.__name__} "
            f"params={sum(p.numel() for p in state['model'].parameters())} "
            f"expander_params={sum(p.numel() for p in expander.parameters()) if expander is not None else 0}",
            flush=True,
        )

    executor = None
    next_epoch_future = None
    if args.cache_mode == "full":
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1)
        next_epoch_future = executor.submit(
            prepare_epoch_batches,
            args.input_h5,
            1,
            args.batch_size,
            args.steps_per_epoch,
            args.sample_fraction,
            args.seed,
            cache_dtype,
            args.pin_cache,
            args.game_order,
            args.max_batch_sentences,
            args.max_view_sentences,
            args.train_game_indices,
        )

    description_rng = np.random.default_rng(args.seed + 704_971)
    try:
        for epoch in range(1, args.epochs + 1):
            for state in arms:
                state["model"].train()
                if state["expander"] is not None:
                    state["expander"].train()
            epoch_sums = {state["args"].arm: {} for state in arms}
            epoch_batches, next_epoch_future = iter_epoch(args, epoch, next_epoch_future, executor, cache_dtype)

            for batch_index, batch in enumerate(epoch_batches, start=1):
                rng_state = capture_rng_state(device)
                description_state = copy.deepcopy(description_rng.bit_generator.state)
                for state in arms:
                    arm_args = state["args"]
                    current_grl = grl_lambda_at(state["global_step"], args.steps_per_epoch, arm_args)
                    state["adversary"].grl.lambda_ = current_grl
                    restore_rng_state(rng_state, device)
                    description_rng.bit_generator.state = copy.deepcopy(description_state)
                    metrics = run_training_batch(
                        batch,
                        state["model"],
                        state["expander"],
                        state["adversary"],
                        state["optimizer"],
                        state["scaler"],
                        arm_args,
                        device,
                        amp_enabled,
                        pin_transfer,
                        description_bank=description_bank,
                        description_rng=description_rng,
                        recommendation_targets_np=recommendation_targets,
                        cache_dtype=cache_dtype,
                    )
                    metrics["grl_lambda"] = current_grl
                    state["global_step"] += 1
                    state["last_metrics"] = metrics
                    sums = epoch_sums[arm_args.arm]
                    for key, value in metrics.items():
                        sums[key] = sums.get(key, 0.0) + value

                    if batch_index == 1 or batch_index % args.log_every == 0:
                        print(
                            f"arm={arm_args.arm} epoch={epoch:03d} "
                            f"step={batch_index:04d}/{args.steps_per_epoch} "
                            f"global={state['global_step']} grl={current_grl:.3f} "
                            f"loss={metrics['loss']:.4f} vic={metrics['vicreg']:.4f} "
                            f"sentences=({metrics['sentences_a']:.0f},{metrics['sentences_b']:.0f}) "
                            f"games={','.join(batch['games'][:3])}",
                            flush=True,
                        )

            for state in arms:
                arm_args = state["args"]
                averaged = {
                    key: value / args.steps_per_epoch
                    for key, value in epoch_sums[arm_args.arm].items()
                }
                state["history_rows"].append({
                    "epoch": epoch,
                    "global_step": state["global_step"],
                    **averaged,
                })
                checkpoint = checkpoint_payload(state, epoch, averaged, arm_args, input_dim)
                if not args.no_save:
                    atomic_torch_save(checkpoint, arm_args.checkpoint_out)
                if averaged["loss"] < state["best_loss"]:
                    state["best_loss"] = averaged["loss"]
                    if not args.no_save:
                        atomic_torch_save(checkpoint, arm_args.best_checkpoint_out)
                write_history(state["history_rows"], arm_args.history_tsv)
                write_manifest(
                    arm_args.manifest_json,
                    "running",
                    arm_args,
                    epoch,
                    state["global_step"],
                    averaged,
                )
                if should_run_probe(epoch, arm_args):
                    run_dual_probe(
                        state["model"],
                        arm_args,
                        device,
                        epoch,
                        state["global_step"],
                        state["probe_rows"],
                    )

        for state in arms:
            write_manifest(
                state["args"].manifest_json,
                "done",
                state["args"],
                args.epochs,
                state["global_step"],
                state["last_metrics"],
            )
    except KeyboardInterrupt:
        for state in arms:
            write_manifest(
                state["args"].manifest_json,
                "interrupted",
                state["args"],
                epoch if "epoch" in locals() else 0,
                state["global_step"],
                state["last_metrics"],
            )
        raise
    except BaseException as exc:
        for state in arms:
            write_manifest(
                state["args"].manifest_json,
                "error",
                state["args"],
                epoch if "epoch" in locals() else 0,
                state["global_step"],
                state["last_metrics"],
                error=f"{type(exc).__name__}: {exc}",
            )
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", default=str(DEFAULT_H5))
    parser.add_argument("--sst-checkpoint", default=str(DEFAULT_SST_CHECKPOINT))
    parser.add_argument("--grl-checkpoint-out", default=str(DEFAULT_HEADS_DIR / "paired_grl_latest.pt"))
    parser.add_argument("--grl-best-checkpoint-out", default=str(DEFAULT_HEADS_DIR / "paired_grl_best.pt"))
    parser.add_argument("--grl-history-tsv", default=str(DEFAULT_HEADS_DIR / "paired_grl_history.tsv"))
    parser.add_argument("--grl-manifest-json", default=str(DEFAULT_HEADS_DIR / "paired_grl_manifest.json"))
    parser.add_argument("--grl-probe-history-tsv", default=str(DEFAULT_HEADS_DIR / "paired_grl_dual_probe.tsv"))
    parser.add_argument("--nogrl-checkpoint-out", default=str(DEFAULT_HEADS_DIR / "paired_nogrl_latest.pt"))
    parser.add_argument("--nogrl-best-checkpoint-out", default=str(DEFAULT_HEADS_DIR / "paired_nogrl_best.pt"))
    parser.add_argument("--nogrl-history-tsv", default=str(DEFAULT_HEADS_DIR / "paired_nogrl_history.tsv"))
    parser.add_argument("--nogrl-manifest-json", default=str(DEFAULT_HEADS_DIR / "paired_nogrl_manifest.json"))
    parser.add_argument("--nogrl-probe-history-tsv", default=str(DEFAULT_HEADS_DIR / "paired_nogrl_dual_probe.tsv"))
    parser.add_argument("--probe-every", type=int, default=0)
    parser.add_argument("--probe-start-epoch", type=int, default=3)
    parser.add_argument("--probe-feature-views", type=int, default=2)
    parser.add_argument("--probe-folds", type=int, default=5)
    parser.add_argument("--probe-sample-fraction", type=float, default=0.6)
    parser.add_argument("--text-variant-dir", default=None)
    parser.add_argument("--text-variant-cache", default=None)
    parser.add_argument("--rebuild-text-variant-cache", action="store_true")
    parser.add_argument("--text-variant-feature-views", type=int, default=4)
    parser.add_argument("--text-variant-sample-fraction", type=float, default=1.0)
    parser.add_argument("--text-variant-local-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--text-variant-embed-batch-size", type=int, default=32)
    parser.add_argument("--text-variant-max-sentences", type=int, default=4096)
    parser.add_argument("--tag-text-split-json", default=None)
    parser.add_argument("--tag-text-train-frac", type=float, default=0.7)
    parser.add_argument("--tag-text-val-frac", type=float, default=0.15)
    parser.add_argument("--tag-text-split-seed", type=int, default=20260627)
    parser.add_argument("--tag-text-threshold-steps", type=int, default=33)
    parser.add_argument("--no-save", action="store_true")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batch-sentences", type=int, default=0)
    parser.add_argument("--max-view-sentences", type=int, default=0)
    parser.add_argument("--sample-fraction", type=float, default=0.6)
    parser.add_argument("--train-game-count", type=int, default=0)
    parser.add_argument("--train-game-seed", type=int, default=20260626)
    parser.add_argument("--train-game-anchor-appids", default="1091500,1385380")
    parser.add_argument("--cache-mode", choices=["queue", "full"], default="queue")
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--pin-cache", action="store_true")
    parser.add_argument("--backward-mode", choices=["recompute", "split_recompute", "standard"], default="recompute")
    parser.add_argument("--game-order", choices=["random", "largest_first", "smallest_first", "file"], default="random")

    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--encoder-arch", choices=["latent_mlp", "hierarchical"], default="latent_mlp")
    parser.add_argument("--output-dim", type=int, default=18)
    parser.add_argument("--reduce-hidden", type=parse_int_list, default=(128, 64, 32))
    parser.add_argument("--probe-hidden", type=int, default=256)
    parser.add_argument("--num-latents", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--vicreg-invariance-weight", type=float, default=25.0)
    parser.add_argument("--vicreg-variance-weight", type=float, default=25.0)
    parser.add_argument("--vicreg-covariance-weight", type=float, default=1.0)
    parser.add_argument("--compact-variance-weight", type=float, default=0.0)
    parser.add_argument("--compact-covariance-weight", type=float, default=0.0)
    parser.add_argument("--vicreg-scope", choices=["game", "slot"], default="game")
    parser.add_argument("--expander-dim", type=int, default=1024)
    parser.add_argument("--expander-hidden", type=parse_int_list, default=(128, 512))
    parser.add_argument("--expander-dropout", type=float, default=0.0)
    parser.add_argument("--description-align-weight", type=float, default=0.0)
    parser.add_argument("--description-mse-weight", type=float, default=0.0)
    parser.add_argument("--description-align-temperature", type=float, default=0.07)
    parser.add_argument("--description-dir", default=str(SCRIPT_DIR / "tags" / "game_descriptions"))
    parser.add_argument("--description-cache", default=str(DEFAULT_DESCRIPTION_CACHE))
    parser.add_argument("--overwrite-description-cache", action="store_true")
    parser.add_argument("--description-include-extra-cases", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--description-max-sentences", type=int, default=512)
    parser.add_argument("--description-embed-batch-size", type=int, default=16)
    parser.add_argument("--description-local-model", default=None)
    parser.add_argument("--recommendation-decorr-weight", type=float, default=0.0)
    parser.add_argument("--grl-recommendation-decorr-weight", type=float, default=0.0)
    parser.add_argument("--nogrl-recommendation-decorr-weight", type=float, default=0.0)
    parser.add_argument("--recommendation-reviews-dir", default=None)
    parser.add_argument("--recommendation-label-min-length", type=int, default=0)
    parser.add_argument("--recommendation-min-label-count", type=int, default=10)
    parser.add_argument("--recommendation-target-transform", choices=["identity", "logit"], default="logit")
    parser.add_argument("--grl-adversary-weight", type=float, default=10.0)
    parser.add_argument("--nogrl-adversary-weight", type=float, default=0.0)
    parser.add_argument("--grl-lambda", type=float, default=1.0)
    parser.add_argument("--grl-warmup-epochs", type=float, default=5.0)
    parser.add_argument("--grl-ramp-epochs", type=float, default=10.0)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
