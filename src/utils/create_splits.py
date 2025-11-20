# create_splits.py

import numpy as np
import pandas as pd
import os

from data_io import load_target  # loads y

# or from data_io import load_expression if you want X-shape instead


def main():
    # --------------------------------------------------------
    # 1. Load the data (to know N)
    # --------------------------------------------------------
    y = load_target()  # length N
    N = len(y)

    # --------------------------------------------------------
    # 2. Random split (fixed seed)
    # --------------------------------------------------------
    indices = np.arange(N)

    rng = np.random.RandomState(42)
    rng.shuffle(indices)

    test_fraction = 0.2
    n_test = int(test_fraction * N)

    test_idx = indices[:n_test]
    trainval_idx = indices[n_test:]

    # --------------------------------------------------------
    # 3. Save into the splits directory
    # --------------------------------------------------------
    OUT_DIR = "data/splits"
    os.makedirs(OUT_DIR, exist_ok=True)

    pd.Series(trainval_idx).to_csv(
        os.path.join(OUT_DIR, "trainval_indices.csv"),
        index=False,
        header=False,
    )

    pd.Series(test_idx).to_csv(
        os.path.join(OUT_DIR, "test_indices.csv"),
        index=False,
        header=False,
    )

    print("Splits created.")
    print(f"Train/val size: {len(trainval_idx)}")
    print(f"Test size:      {len(test_idx)}")


if __name__ == "__main__":
    main()
