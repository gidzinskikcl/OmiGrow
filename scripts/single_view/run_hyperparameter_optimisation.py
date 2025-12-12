import os
import argparse
import sys
import json

import optuna

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
)
from training.single_view_tuning import make_objective
from utils.scaler import standardise


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
        description="Run single-view hyperparameter tuning for a given modality."
    )
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        help="Which modality to tune: 'expr' or 'prot'.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="S1",
        help="Model identifier (e.g. S1, S2). Used only for naming results.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="Root directory for saving tuning results.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna trials to run.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of CV folds.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=300,
        help="Max epochs per training run.",
    )

    args = parser.parse_args()
    modality = args.modality.lower()
    model_id = f"{args.model_id}"
    results_root = args.results_root
    n_trials = args.n_trials
    n_splits = args.n_splits
    max_epochs = args.max_epochs
    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------
    load_X = get_loader_for_modality(modality)
    X = load_X()
    y = load_target()
    trainval_idx = load_trainval_indices()
    check_dimensions(X, y)
    # --------------------------------------------------------
    # 2. Standardise X using train indices only
    # --------------------------------------------------------
    X_scaled = standardise(X, trainval_idx)
    # --------------------------------------------------------
    # 4. Prepare output dir
    # --------------------------------------------------------
    output_dir = prepare_output_dir(results_root, model_id, modality)
    # --------------------------------------------------------
    # 5. Run optimisation
    # --------------------------------------------------------
    study_name = f"{model_id}_{modality}"
    sampler = optuna.samplers.TPESampler(seed=123)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
    )
    objective = make_objective(X_scaled, y, trainval_idx, n_splits, max_epochs)
    study.optimize(objective, n_trials=n_trials)
    # --------------------------------------------------------
    # 6. Save results
    # --------------------------------------------------------
    # All trials as a CSV (similar to your old tuning_results_*.csv)
    trials_df = study.trials_dataframe()
    trials_csv = os.path.join(output_dir, "optuna_trials.csv")
    trials_df.to_csv(trials_csv, index=False)

    # Best params as JSON
    best_params_path = os.path.join(output_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)

    # Best value (e.g., RMSE) as a small text file
    best_value_path = os.path.join(output_dir, "best_value.txt")
    with open(best_value_path, "w") as f:
        f.write(str(study.best_value))

    print(f"Finished Optuna tuning for modality={modality}")
    print(f"Best RMSE: {study.best_value:.6f}")
    print(f"Best params saved to: {best_params_path}")
    print(f"All trials saved to:  {trials_csv}")


if __name__ == "__main__":
    main()
