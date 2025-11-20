import argparse
import os
import sys
import random

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import backend as K
import gc

# Add src/ to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project_root
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utils.data_io import (
    get_loader_for_modality,
    load_target,
    load_train_indices,
    load_test_indices,
    load_best_params,
)

from training.train_single_view import train

# Fixed, published seeds for reproducibility
SEEDS = [3, 7, 11, 19, 23, 31, 42, 57, 73, 101]


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
        default="results",
        help="Root directory for saving training results.",
    )

    args = parser.parse_args()
    modality = args.modality.lower()
    model_id = f"{args.model_id}"

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    load_X = get_loader_for_modality(modality=modality)
    X = load_X()
    y = load_target()
    train_idx = load_train_indices()
    test_idx = load_test_indices()

    if X.shape[0] != len(y):
        raise ValueError(
            f"X and y have different number of samples: {X.shape[0]} vs {len(y)}"
        )

    best_params = load_best_params(model_id=f"{model_id}_{modality}")
    print("Using best hyperparameters from JSON:")
    print(best_params)
    # --------------------------------------------------------
    # Standardise X
    # --------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = X.astype(np.float32, copy=True)
    X_scaled[train_idx] = scaler.fit_transform(X_scaled[train_idx])

    mask_other = np.ones(len(X_scaled), dtype=bool)
    mask_other[train_idx] = False
    if mask_other.any():
        X_scaled[mask_other] = scaler.transform(X_scaled[mask_other])

    # --------------------------------------------------------
    # Run training for multiple seeds
    # --------------------------------------------------------
    output_dir = os.path.join(args.results_root, f"trained_{model_id}_{modality}")
    os.makedirs(output_dir, exist_ok=True)

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
            batch_size=256,
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
    best_row = metrics_df.loc[metrics_df["MAE"].idxmin()]
    best_seed = int(best_row["seed"])
    print(f"\nBest seed by MAE: {best_seed}")
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
        batch_size=256,
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
    os.makedirs("models", exist_ok=True)
    weights_path = os.path.join(
        "models", f"{model_id}_{modality}_{best_seed}.weights.h5"
    )
    final_model.save_weights(weights_path)
    print("Saved model weights to:", weights_path)

    del final_model
    K.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
