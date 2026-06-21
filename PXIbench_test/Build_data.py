"""End-to-end dataset builder for the PXI benchmark experiments.

Runs the four build steps in order and produces everything
``test_latent_query_model.py`` needs to train, using only its defaults:

    1. download_benchmarks       -> PXIbenchmark_data/  (incl. benchmark.csv)
    2. generate_pseudo_text      -> pesudo_data/pseudo_text_data_<tag>.csv
    3. embed_pseudo_text_sentences -> pesudo_data/pseudo_text_sentence_embeddings<_tag>/
    4. build_h5                  -> pesudo_data/benchmark_sentence_latent_query<_tag>.h5

Each step is skipped if its primary output already exists, so the script is
idempotent and resumable. After it finishes you can launch:

    python PXIbench_test/test_latent_query_model.py

Common flags:
    --variants-per-row N     1 builds the `one_per_game` dataset; >1 builds the
                             `multi` dataset that test_latent_query_model.py
                             defaults to. Default: 5 (matches existing data).
    --skip STEP [...]        Skip one or more of: download, generate, embed, h5
    --only STEP [...]        Run only these steps.
    --force                  Re-run all steps even if outputs already exist.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_DIR = SCRIPT_DIR / "build"
PXI_DIR = SCRIPT_DIR / "PXIbenchmark_data"
PESUDO_DIR = SCRIPT_DIR / "pesudo_data"

STEPS = ("download", "generate", "embed", "h5")


def run(name, args):
    """Run a build script as a subprocess so each stage gets a clean argparse."""
    print(f"\n===== [{name}] {' '.join(map(str, args))} =====", flush=True)
    result = subprocess.run([sys.executable, "-u", *map(str, args)])
    if result.returncode != 0:
        raise SystemExit(f"[{name}] failed with exit code {result.returncode}")


def needs(path, force):
    return force or not Path(path).exists()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants-per-row", type=int, default=5,
                        help="1 -> 'one_per_game' dataset; >1 -> 'multi' dataset (default 5).")
    parser.add_argument("--only", nargs="+", choices=STEPS, default=None)
    parser.add_argument("--skip", nargs="+", choices=STEPS, default=[])
    parser.add_argument("--force", action="store_true",
                        help="Re-run every selected step even if its output exists.")
    # download
    parser.add_argument("--download-limit", type=int, default=None,
                        help="Only download the first N games (for quick smoke tests).")
    # embed
    parser.add_argument("--embed-device", default=None,
                        help="e.g. 'cuda' or 'cpu' (passed to embed_pseudo_text_sentences).")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--embed-backend", choices=["transformers", "sentence-transformers"],
                        default="transformers")
    return parser.parse_args()


def main():
    args = parse_args()
    run_set = set(args.only) if args.only else set(STEPS)
    run_set -= set(args.skip)

    tag = "_multi" if args.variants_per_row > 1 else "_one_per_game"
    pseudo_csv = PESUDO_DIR / f"pseudo_text_data{tag}.csv"
    embed_dir = PESUDO_DIR / f"pseudo_text_sentence_embeddings{tag.replace('_one_per_game','')}"
    # ^^ embed dir uses '_multi' suffix or no suffix at all (the existing convention)
    h5_path = PESUDO_DIR / f"benchmark_sentence_latent_query{tag.replace('_one_per_game','')}.h5"
    benchmark_csv = PXI_DIR / "benchmark.csv"
    metadata_csv = embed_dir / "sentence_metadata.csv"
    embeddings_npy = embed_dir / "sentence_embeddings.npy"

    PXI_DIR.mkdir(parents=True, exist_ok=True)
    PESUDO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"PXIbench build pipeline | variants_per_row={args.variants_per_row} (tag='{tag}')")
    print(f"  benchmark.csv      -> {benchmark_csv}")
    print(f"  pseudo text csv    -> {pseudo_csv}")
    print(f"  sentence embed dir -> {embed_dir}")
    print(f"  h5 dataset         -> {h5_path}")
    print(f"  run steps: {sorted(run_set)}")

    # 1. download_benchmarks -> PXIbenchmark_data/benchmark.csv
    if "download" in run_set and needs(benchmark_csv, args.force):
        download_args = [BUILD_DIR / "download_benchmarks.py", "--out", PXI_DIR]
        if args.download_limit:
            download_args += ["--limit", args.download_limit]
        if args.force:
            download_args += ["--overwrite"]
        run("download_benchmarks", download_args)
        # download writes final_benchmark_values_wide.csv; the pipeline expects benchmark.csv.
        wide_csv = PXI_DIR / "final_benchmark_values_wide.csv"
        if wide_csv.exists() and not benchmark_csv.exists():
            shutil.copyfile(wide_csv, benchmark_csv)
            print(f"  copied {wide_csv.name} -> {benchmark_csv.name}")
    elif "download" in run_set:
        print(f"[download] skip: {benchmark_csv.name} already exists")

    # 2. generate_pseudo_text
    if "generate" in run_set and needs(pseudo_csv, args.force):
        run("generate_pseudo_text", [
            BUILD_DIR / "generate_pseudo_text.py",
            "--input", benchmark_csv,
            "--output", pseudo_csv,
            "--variants-per-row", args.variants_per_row,
        ])
    elif "generate" in run_set:
        print(f"[generate] skip: {pseudo_csv.name} already exists")

    # 3. embed_pseudo_text_sentences
    if "embed" in run_set and needs(embeddings_npy, args.force):
        embed_args = [
            BUILD_DIR / "embed_pseudo_text_sentences.py",
            "--input-csv", pseudo_csv,
            "--output-dir", embed_dir,
            "--batch-size", args.embed_batch_size,
            "--backend", args.embed_backend,
        ]
        if args.embed_device:
            embed_args += ["--device", args.embed_device]
        run("embed_pseudo_text_sentences", embed_args)
    elif "embed" in run_set:
        print(f"[embed] skip: {embeddings_npy.name} already exists")

    # 4. build_h5
    if "h5" in run_set and needs(h5_path, args.force):
        run("build_h5", [
            BUILD_DIR / "build_h5.py",
            "--target-csv", pseudo_csv,
            "--sentence-metadata-csv", metadata_csv,
            "--embeddings-npy", embeddings_npy,
            "--output-h5", h5_path,
        ])
    elif "h5" in run_set:
        print(f"[h5] skip: {h5_path.name} already exists")

    print(f"\nDone. Train with:\n  python {SCRIPT_DIR / 'test_latent_query_model.py'}")
    if args.variants_per_row == 1:
        print(f"  (pass --input-h5 {h5_path} since the default expects the multi dataset)")


if __name__ == "__main__":
    main()
