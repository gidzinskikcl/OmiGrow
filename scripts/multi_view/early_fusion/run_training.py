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

# Go up four levels: run_dual_view_early_fusion_training.py
# -> early_fusion -> multi_view -> scripts -> project_root
ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)  # project_root
SRC = os.path.join(ROOT, "src")

# Make src/ importable
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utils.data_io import (
    get_loader_for_modality,
    load_target,
    load_trainval_indices,
    load_test_indices,
)

from training.single_view_training import train
from utils.scaler import standardise


# Fixed, published seeds for reproducibility
SEEDS = [3, 7, 11, 19, 23, 31, 42, 57, 73, 101]


def check_dimensions(X1, X2, y):
    if X1.shape[0] != y.shape[0] or X2.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have different number of samples: "
            f"{X1.shape[0]} | {X2.shape[0]} vs {y.shape[0]}"
        )


def prepare_output_dir(results_root, model_id, modality_1, modality_2):
    output_dir = os.path.join(
        results_root,
        f"training_{model_id}_{modality_1}_{modality_2}",
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Run early-fusion dual-view training.")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier (e.g. EF1, EF2).",
    )
    parser.add_argument(
        "--modality_1",
        type=str,
        required=True,
        help="First modality to use (e.g. 'expr', 'prot').",
    )
    parser.add_argument(
        "--modality_2",
        type=str,
        required=True,
        help="Second modality to use (e.g. 'expr', 'prot').",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="Root directory for saving training results.",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        required=True,
        help="File path to JSON with best hyperparameters.",
    )

    args = parser.parse_args()
    modality_1 = args.modality_1.lower()
    modality_2 = args.modality_2.lower()
    model_id = args.model_id
    results_root = args.results_root
    params_path = args.params_path
    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    load_X1 = get_loader_for_modality(modality_1)
    load_X2 = get_loader_for_modality(modality_2)
    X1 = load_X1().astype(np.float32)  # shape: (n_samples, d1)
    X2 = load_X2().astype(np.float32)  # shape: (n_samples, d2)
    y_arr = load_target().astype(np.float32)  # shape: (n_samples,)

    trainval_idx = load_trainval_indices()
    test_idx = load_test_indices()
    check_dimensions(X1, X2, y_arr)
    # --------------------------------------------------------
    # 2. Load best hyperparameters
    # --------------------------------------------------------
    # from params_path JSON file
    with open(params_path, "r") as f:
        best_params = json.load(f)
    print("Using best hyperparameters from JSON:")
    print(best_params)
    # --------------------------------------------------------
    # Early fusion = concatenate along feature axis
    # --------------------------------------------------------
    # shapes: (n_samples, d1), (n_samples, d2) -> (n_samples, d1 + d2)
    X_early = np.concatenate([X1, X2], axis=1)
    # --------------------------------------------------------
    # Standardise X
    # --------------------------------------------------------
    X_scaled = standardise(X_early, trainval_idx)
    # --------------------------------------------------------
    # Prepare output dir
    # --------------------------------------------------------
    output_dir = prepare_output_dir(results_root, model_id, modality_1, modality_2)
    # --------------------------------------------------------
    # Run training for multiple seeds
    # --------------------------------------------------------
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
            y=y_arr,
            trainval_idx=trainval_idx,
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
        y=y_arr,
        trainval_idx=trainval_idx,
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
        "weights", f"{model_id}_{modality_1}_{modality_2}_{best_seed}.weights.h5"
    )
    final_model.save_weights(weights_path)
    print("Saved model weights to:", weights_path)

    del final_model
    K.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
