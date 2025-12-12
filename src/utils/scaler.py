import numpy as np
from sklearn.preprocessing import StandardScaler


def standardise(X: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    X_scaled = X.astype(np.float32, copy=True)
    # fit on train subset
    X_scaled[train_idx] = scaler.fit_transform(X_scaled[train_idx])

    # transform the rest (non-train indices) consistently
    mask_other = np.ones(len(X_scaled), dtype=bool)
    mask_other[train_idx] = False
    if mask_other.any():
        X_scaled[mask_other] = scaler.transform(X_scaled[mask_other])

    return X_scaled
