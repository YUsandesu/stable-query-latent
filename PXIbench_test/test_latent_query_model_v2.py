"""Thin entry point that trains the v2 architecture
(latent_query_model_v2.LatentQueryFunnelRegressor) by reusing the exact training
loop in test_latent_query_model.py — only the model and the default output paths
differ, so v1 vs v2 is an apples-to-apples comparison.

    python PXIbench_test/test_latent_query_model_v2.py [same flags as the v1 trainer]
"""

import sys

import test_latent_query_model as base


def main():
    # Default to the v2 architecture and v2 output filenames unless the user
    # overrode them on the command line.
    if not any(arg == "--model" or arg.startswith("--model=") for arg in sys.argv[1:]):
        sys.argv += ["--model", "v2"]

    heads_dir = base.SCRIPT_DIR / "heads"
    defaults = {
        "--model-out": str(heads_dir / "latent_query_benchmark_multi_classifier_v2.pt"),
        "--history-txt": str(heads_dir / "latent_query_training_history_multi_classifier_v2.txt"),
        "--per-dim-txt": str(heads_dir / "latent_query_per_dim_multi_classifier_v2.txt"),
    }
    for flag, value in defaults.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:]):
            sys.argv += [flag, value]

    base.main()


if __name__ == "__main__":
    main()
