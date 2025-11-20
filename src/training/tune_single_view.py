from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
)
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras import backend as K
import gc

from training import single_view


def tune(
    X,
    y,
    trainval_idx,
    grid,
    n_splits=5,
    max_epochs=300,
    batch_size=256,
):
    """
    X, y: full dataset (numpy arrays or pandas .values)
    trainval_idx: indices used for hyperparameter tuning (no test leakage)
    grid: iterable of dicts (e.g. list(ParameterGrid(...)))

    Returns
    -------
    results_df : pd.DataFrame
        One row per hyperparameter setting with mean metrics across folds:
        mean_MAE, mean_RMSE, mean_MDAE, mean_PCC.
    """
    X_tv = X[trainval_idx]
    y_tv = y[trainval_idx]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

    # --------------------------------------------------------
    # Pre-filter grid to know how many fits in total
    # --------------------------------------------------------
    valid_params = []
    for params in grid:
        if params["optimizer"] == "adam" and params["learning_rate"] > 0.005:
            continue
        if params["optimizer"] == "sgd" and params["learning_rate"] < 0.01:
            continue
        valid_params.append(params)

    n_configs = len(valid_params)
    total_fits = n_configs * n_splits
    fits_done = 0

    # --------------------------------------------------------
    # Tuning
    # --------------------------------------------------------

    print(f"Total valid configs: {n_configs}")
    print(f"Total model fits:    {total_fits} (configs × folds)")

    all_results = []

    for i, params in enumerate(valid_params, start=1):
        print(f"\n=== Config {i}/{n_configs}: {params} ===")

        fold_maes = []
        fold_rmses = []
        fold_mdaes = []
        fold_pccs = []

        # --------------------------------------------------------
        # Cross-validation
        # --------------------------------------------------------

        for fold, (train_idx_local, val_idx_local) in enumerate(
            kf.split(X_tv), start=1
        ):
            fits_done += 1
            print(
                f"  [Fit {fits_done}/{total_fits}] "
                f"Config {i}/{n_configs}, Fold {fold}/{n_splits}"
            )

            X_tr = X_tv[train_idx_local]
            y_tr = y_tv[train_idx_local]
            X_val = X_tv[val_idx_local]
            y_val = y_tv[val_idx_local]

            model = single_view.build(
                input_dim=X_tv.shape[1],
                hidden_layers=params["hidden_layers"],
                neurons=params["neurons"],
                learning_rate=params["learning_rate"],
                optimizer_name=params["optimizer"],
                dropout=params["dropout"],
                kernel_constraint=params["kernel_constraint"],
            )

            es = EarlyStopping(
                monitor="val_loss",
                patience=25,
                restore_best_weights=True,
                verbose=0,
            )
            ton = TerminateOnNaN()

            model.fit(
                X_tr,
                y_tr,
                validation_data=(X_val, y_val),
                epochs=max_epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=[es, ton],
            )

            y_val_pred = model.predict(X_val, verbose=0).ravel()

            # Guard against non-finite predictions
            if not np.all(np.isfinite(y_val_pred)):
                print("  WARNING: non-finite predictions; penalising config.")
                mae = rmse = mdae = 1e9
                pcc = np.nan
            else:
                mae = mean_absolute_error(y_val, y_val_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                mdae = median_absolute_error(y_val, y_val_pred)

                if np.std(y_val) == 0 or np.std(y_val_pred) == 0:
                    pcc = np.nan
                else:
                    pcc, _ = pearsonr(y_val, y_val_pred)

            fold_maes.append(mae)
            fold_rmses.append(rmse)
            fold_mdaes.append(mdae)
            fold_pccs.append(pcc)

            # Free TF/Keras resources for this model
            K.clear_session()
            gc.collect()

        mean_mae = float(np.mean(fold_maes))
        mean_rmse = float(np.mean(fold_rmses))
        mean_mdae = float(np.mean(fold_mdaes))
        mean_pcc = float(np.nanmean(fold_pccs))

        result_row = {
            **params,
            "mean_MAE": mean_mae,
            "mean_RMSE": mean_rmse,
            "mean_MDAE": mean_mdae,
            "mean_PCC": mean_pcc,
        }
        all_results.append(result_row)

    results_df = pd.DataFrame(all_results)
    return results_df
