# scripts/run_tuning_S.py

import os
import argparse
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project_root
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utils.grid import grid as full_grid
from utils.data_io import (
    get_loader_for_modality,
    load_target,
    load_train_indices,
)
from training.tune_single_view import tune


def main():
    parser = argparse.ArgumentParser(
        description="Run single-view hyperparameter tuning for a given modality."
    )
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        help="Which modality to tune: 'expr', 'prot', or 'flux'.",
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
        help="Root directory for saving tuning results.",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start index of grid slice (inclusive)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index of grid slice (exclusive). Default: full grid.",
    )

    args = parser.parse_args()
    modality = args.modality.lower()

    # ------------------------
    # 1. Slice the grid
    # ------------------------
    # Resolve actual end index if None
    actual_end = args.end if args.end is not None else len(full_grid)
    grid_slice = full_grid[args.start : actual_end]
    grid_size = len(grid_slice)

    if grid_size == 0:
        raise ValueError("Selected grid slice is empty.")

    # Update model ID to ID_{grid_size}
    model_id = f"{args.model_id}_ID_{grid_size}"

    # --------------------------------------------------------
    # 2. Load data
    # --------------------------------------------------------
    load_X = get_loader_for_modality(modality)
    X = load_X()
    y = load_target()
    train_idx = load_train_indices()

    if X.shape[0] != len(y):
        raise ValueError(
            f"X and y have different number of samples: {X.shape[0]} vs {len(y)}"
        )

    # --------------------------------------------------------
    # 3. Standardise X using train indices only
    # --------------------------------------------------------
    scaler = StandardScaler()

    X_scaled = X.astype(np.float32, copy=True)
    # fit on train subset
    X_scaled[train_idx] = scaler.fit_transform(X_scaled[train_idx])

    # transform the rest (non-train indices) consistently
    mask_other = np.ones(len(X_scaled), dtype=bool)
    mask_other[train_idx] = False
    if mask_other.any():
        X_scaled[mask_other] = scaler.transform(X_scaled[mask_other])

    # --------------------------------------------------------
    # 4. Prepare output dir
    # --------------------------------------------------------
    output_dir = os.path.join(
        args.results_root,
        f"results_{model_id}_{modality}",
    )
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # 5. Run hyperparameter tuning
    # --------------------------------------------------------
    results_df = tune(
        X=X_scaled,
        y=y,
        trainval_idx=train_idx,
        grid=grid_slice,
        n_splits=5,
        max_epochs=300,
        batch_size=256,
    )

    # --------------------------------------------------------
    # 6. Save tuning results for this slice
    # --------------------------------------------------------
    # File name includes slice indices to avoid overwriting
    out_csv = os.path.join(output_dir, f"tuning_results_{args.start}_{actual_end}.csv")
    results_df.to_csv(out_csv, index=False)

    print(f"Finished tuning slice {args.start}:{actual_end} ({grid_size} configs)")
    print(f"Saved to: {out_csv}")


if __name__ == "__main__":
    main()
