import argparse
import json
import os
import sys
import optuna


# Go up four levels: run_dual_view_early_fusion_tuning.py
# -> late_fusion -> multi_view -> scripts -> project_root
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
)
from training.late_fusion_tuning import make_objective
from utils.scaler import standardise


def check_dimensions(X1, X2, y):
    if X1.shape[0] != len(y) or X2.shape[0] != len(y):
        raise ValueError(
            f"X and y have different number of samples: {X1.shape[0]} |  {X2.shape[0]} vs {len(y)}"
        )


def prepare_output_dir(results_root, model_id, modality_1, modality_2):
    output_dir = os.path.join(
        results_root,
        f"training_{model_id}_{modality_1}_{modality_2}",
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_parameters(params_path):
    with open(params_path, "r") as f:
        best_params = json.load(f)
    print("Using best hyperparameters from JSON:")
    print(best_params)
    return best_params


def main():
    parser = argparse.ArgumentParser(
        description="Run middle-fusion dual-view hyperparameter tuning."
    )
    parser.add_argument(
        "--model_id", type=str, required=True, help="Model identifier (e.g. MF1, MF2)."
    )
    parser.add_argument(
        "--modality_1",
        type=str,
        required=True,
        help="First modality to use (e.g. 'expr', 'prot').",
    )
    parser.add_argument(
        "--modality_1_params_path",
        type=str,
        required=True,
        help="Path to JSON file with pretrained encoder params for modality 1.",
    )
    parser.add_argument(
        "--modality_2",
        type=str,
        required=True,
        help="Second modality to use (e.g. 'expr', 'prot').",
    )
    parser.add_argument(
        "--modality_2_params_path",
        type=str,
        required=True,
        help="Path to JSON file with pretrained encoder params for modality 2.",
    )
    parser.add_argument(
        "--weights_1_path",
        type=str,
        required=True,
        help="Path to pretrained encoder weights for modality 1.",
    )
    parser.add_argument(
        "--weights_2_path",
        type=str,
        required=True,
        help="Path to pretrained encoder weights for modality 2.",
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
    model_id = args.model_id
    modality_1 = args.modality_1.lower()
    modality_1_params_path = args.modality_1_params_path
    weights_1_path = args.weights_1_path
    modality_2 = args.modality_2.lower()
    modality_2_params_path = args.modality_2_params_path
    weights_2_path = args.weights_2_path
    results_root = args.results_root
    n_trials = args.n_trials
    n_splits = args.n_splits
    max_epochs = args.max_epochs
    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------
    load_X1 = get_loader_for_modality(modality_1)
    load_X2 = get_loader_for_modality(modality_2)
    X1 = load_X1()  # shape: (n_samples, d1)
    X2 = load_X2()  # shape: (n_samples, d2)
    y = load_target()  # shape: (n_samples,)
    trainval_idx = load_trainval_indices()
    check_dimensions(X1, X2, y)
    # --------------------------------------------------------
    # 2. Standardise X using train indices only
    # --------------------------------------------------------
    X1_scaled = standardise(X=X1, train_idx=trainval_idx)
    X2_scaled = standardise(X=X2, train_idx=trainval_idx)
    # --------------------------------------------------------
    # 4. Prepare output dir
    # --------------------------------------------------------
    output_dir = prepare_output_dir(results_root, model_id, modality_1, modality_2)
    # --------------------------------------------------------
    # 5. Load pretrained encoder params
    # --------------------------------------------------------
    X1_params = load_parameters(modality_1_params_path)
    X2_params = load_parameters(modality_2_params_path)
    # --------------------------------------------------------
    # 6. Run optimisation
    # --------------------------------------------------------
    study_name = f"{model_id}_{modality_1}_{modality_2}"
    sampler = optuna.samplers.TPESampler(seed=123)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
    )
    objective = make_objective(
        X1_scaled=X1_scaled,
        X2_scaled=X2_scaled,
        X1_weights_path=weights_1_path,
        X2_weights_path=weights_2_path,
        X1_params=X1_params,
        X2_params=X2_params,
        y=y,
        trainval_idx=trainval_idx,
        n_splits=n_splits,
        max_epochs=max_epochs,
    )

    study.optimize(objective, n_trials=n_trials)
    # --------------------------------------------------------
    # 7. Save results
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

    print(f"Finished Optuna tuning for middle fusion of: {modality_1}+{modality_2}")
    print(f"Best RMSE: {study.best_value:.6f}")
    print(f"Best params saved to: {best_params_path}")
    print(f"All trials saved to:  {trials_csv}")


if __name__ == "__main__":
    main()
