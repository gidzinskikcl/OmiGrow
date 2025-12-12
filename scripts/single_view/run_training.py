import argparse
import json
import os
import sys
import random

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
import gc

# Go up three levels: single_view → scripts → project_root
ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # project_root
SRC = os.path.join(ROOT, "src")

# Add both project root and src to sys.path (handy for other imports too)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utils.data_io import (
    get_loader_for_modality,
    load_target,
    load_trainval_indices,
    load_test_indices,
)
from utils.scaler import standardise


from training.single_view_training import train

# Fixed, published seeds for reproducibility
SEEDS = [3, 7, 11, 19, 23, 31, 42, 57, 73, 101]


def check_dimensions(X, y):
    if X.shape[0] != len(y):
        raise ValueError(
            f"X and y have different number of samples: {X.shape[0]} vs {len(y)}"
        )


def prepare_output_dir(results_root, model_id, modality):
    output_dir = os.path.join(
        results_root,
        f"training_{model_id}_{modality}",
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run single-view training for a given modality."
    )
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        help="Which modality to train: 'expr', 'prot', or 'flux'.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="S1",
        help="Model identifier (e.g. S1, S2, S3). Used only for naming results.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results/predictions",
        help="Root directory for saving training results.",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        required=True,
        help="File path to JSON with best hyperparameters.",
    )

    args = parser.parse_args()
    modality = args.modality.lower()
    model_id = f"{args.model_id}"
    results_root = args.results_root
    params_path = args.params_path
    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------
    load_X = get_loader_for_modality(modality=modality)
    X = load_X()
    y = load_target()
    train_idx = load_trainval_indices()
    test_idx = load_test_indices()
    check_dimensions(X, y)
    # --------------------------------------------------------
    # 2. Load best hyperparameters
    # --------------------------------------------------------
    # from params_path JSON file
    with open(params_path, "r") as f:
        best_params = json.load(f)
    print("Using best hyperparameters from JSON:")
    print(best_params)
    # --------------------------------------------------------
    # 3. Standardise X
    # --------------------------------------------------------
    X_scaled = standardise(X, train_idx)
    # --------------------------------------------------------
    # Run training for multiple seeds
    # --------------------------------------------------------
    output_dir = prepare_output_dir(results_root, model_id, modality)

    all_metrics = []

    for seed in SEEDS:
        print("\n" + "=" * 60)
        print(f"Running training with seed {seed}...")
        print("=" * 60)

        # Set seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)

        model, history, metrics = train(
            X=X_scaled,
            y=y,
            trainval_idx=train_idx,
            test_idx=test_idx,
            best_params=best_params,
            max_epochs=300,
        )

        metrics_with_seed = dict(metrics)
        metrics_with_seed["seed"] = seed
        all_metrics.append(metrics_with_seed)

        # Free TF/Keras resources for this run
        del model
        K.clear_session()
        gc.collect()

    # --------------------------------------------------------
    # Save metrics and weights
    # --------------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv_path = os.path.join(output_dir, "test_metrics_seeds.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print("Saved test metrics to:", metrics_csv_path)

    summary = metrics_df.drop(columns=["seed"]).agg(["mean", "std"])
    summary_path = os.path.join(output_dir, "test_metrics_summary.csv")
    summary.to_csv(summary_path)
    print("Saved summary (mean/std) to:", summary_path)

    # --------------------------------------------------------
    # Retrain once with best seed and save that single model
    # --------------------------------------------------------
    best_row = metrics_df.loc[metrics_df["RMSE"].idxmin()]
    best_seed = int(best_row["seed"])
    print(f"\nBest seed by RMSE: {best_seed}")
    print("Retraining final model with this seed and saving weights...")

    np.random.seed(best_seed)
    random.seed(best_seed)
    tf.random.set_seed(best_seed)

    final_model, final_history, final_metrics = train(
        X=X_scaled,
        y=y,
        trainval_idx=train_idx,
        test_idx=test_idx,
        best_params=best_params,
        max_epochs=300,
    )

    # Save final metrics
    final_metrics_path = os.path.join(
        output_dir, f"{best_seed}_final_test_metrics.json"
    )
    pd.Series(final_metrics | {"seed": best_seed}).to_json(final_metrics_path)
    print("Saved final test metrics to:", final_metrics_path)

    # Save final training history
    history_df = pd.DataFrame(final_history.history)
    history_csv_path = os.path.join(
        output_dir, f"{best_seed}_final_training_history.csv"
    )
    history_df.to_csv(history_csv_path, index=False)
    print("Saved final training history to:", history_csv_path)

    # Save final weights
    os.makedirs("weights", exist_ok=True)
    weights_path = os.path.join(
        "weights", f"{model_id}_{modality}_{best_seed}.weights.h5"
    )
    final_model.save_weights(weights_path)
    print("Saved model weights to:", weights_path)

    del final_model
    K.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
