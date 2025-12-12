from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
)
from scipy.stats import pearsonr
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

from models import encoders


def train(
    X1,
    X2,
    y,
    trainval_idx,
    test_idx,
    best_params,
    X1_weights_path,
    X2_weights_path,
    X1_params,
    X2_params,
    max_epochs=300,
):
    # Split data
    X1_tv = X1[trainval_idx]
    X2_tv = X2[trainval_idx]
    y_tv = y[trainval_idx]
    X1_test = X1[test_idx]
    X2_test = X2[test_idx]
    y_test = y[test_idx]

    model_1 = encoders.load_single_view_model(
        input_dim=X1.shape[1],
        weights_path=X1_weights_path,
        params=X1_params,
    )
    model_2 = encoders.load_single_view_model(
        input_dim=X2.shape[1],
        weights_path=X2_weights_path,
        params=X2_params,
    )

    # Early stopping on validation loss
    es = EarlyStopping(
        monitor="val_loss",
        patience=25,
        restore_best_weights=True,
        verbose=1,
    )
    ton = TerminateOnNaN()

    # Train single-view models on their own inputs
    history_1 = model_1.fit(
        X1_tv,
        y_tv,
        validation_data=(X1_test, y_test),
        epochs=max_epochs,
        batch_size=X1_params["batch_size"],
        verbose=0,
        callbacks=[es, ton],
    )

    history_2 = model_2.fit(
        X2_tv,
        y_tv,
        validation_data=(X2_test, y_test),
        epochs=max_epochs,
        batch_size=X2_params["batch_size"],
        verbose=0,
        callbacks=[es, ton],
    )

    # Predictions on validation folds
    y_val_pred_1 = model_1.predict(X1_test, verbose=0).ravel()
    y_val_pred_2 = model_2.predict(X2_test, verbose=0).ravel()

    alpha = best_params["alpha"]
    y_pred = alpha * y_val_pred_1 + (1.0 - alpha) * y_val_pred_2

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mdae = median_absolute_error(y_test, y_pred)

    # Pearson correlation (PCC)
    if np.std(y_test) == 0 or np.std(y_pred) == 0:
        pcc = np.nan
    else:
        pcc, _ = pearsonr(y_test, y_pred)

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MDAE": float(mdae),
        "PCC": float(pcc) if np.isfinite(pcc) else np.nan,
    }

    return model_1, model_2, history_1, history_2, metrics
