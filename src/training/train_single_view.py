from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
)
from scipy.stats import pearsonr
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

from training import single_view


def train(
    X,
    y,
    trainval_idx,
    test_idx,
    best_params,
    max_epochs=300,
    batch_size=256,
):
    """
    Train final single-view model on train set and evaluate on test set.

    Parameters
    ----------
    X : np.ndarray
        Full feature matrix (already preprocessed / scaled).
    y : np.ndarray
        Full target vector.
    train_idx : np.ndarray
        Indices of training samples.
    test_idx : np.ndarray
        Indices of test samples.
    best_params : dict
        Hyperparameters: hidden_layers, neurons, learning_rate,
        optimizer, dropout, kernel_constraint.
    """
    # Split data
    X_tv = X[trainval_idx]
    y_tv = y[trainval_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    # Build model with best hyperparameters
    model = single_view.build(
        input_dim=X_tv.shape[1],
        hidden_layers=best_params["hidden_layers"],
        neurons=best_params["neurons"],
        learning_rate=best_params["learning_rate"],
        optimizer_name=best_params["optimizer"],
        dropout=best_params["dropout"],
        kernel_constraint=best_params["kernel_constraint"],
    )

    # Early stopping on validation loss
    es = EarlyStopping(
        monitor="val_loss",
        patience=25,
        restore_best_weights=True,
        verbose=1,
    )
    ton = TerminateOnNaN()

    history = model.fit(
        X_tv,
        y_tv,
        validation_split=0.1,  # small internal val for early stopping only
        epochs=max_epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[es, ton],
    )

    # Test evaluation
    y_pred = model.predict(X_test, verbose=0).ravel()

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

    return model, history, metrics
