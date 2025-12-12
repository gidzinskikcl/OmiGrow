from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
)
from scipy.stats import pearsonr
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

from models import encoders, middle_fusion


def train(
    X1,
    X2,
    y,
    trainval_idx,
    test_idx,
    best_params,
    weights_1_path,
    weights_2_path,
    params_1,
    params_2,
    trainable: bool = False,
    max_epochs=300,
):
    # Split data
    X1_tv = X1[trainval_idx]
    X2_tv = X2[trainval_idx]
    y_tv = y[trainval_idx]
    X1_test = X1[test_idx]
    X2_test = X2[test_idx]
    y_test = y[test_idx]

    encoder_1 = encoders.load_pretrained_encoder(
        input_dim=X1.shape[1],
        weights_path=weights_1_path,
        params=params_1,
        trainable=trainable,
    )
    encoder_2 = encoders.load_pretrained_encoder(
        input_dim=X2.shape[1],
        weights_path=weights_2_path,
        params=params_2,
        trainable=trainable,
    )

    # Build model with best hyperparameters
    model = middle_fusion.build(
        input_1_dim=X1_test.shape[1],
        input_2_dim=X2_test.shape[1],
        encoder_1=encoder_1,
        encoder_2=encoder_2,
        hidden_layers=best_params["n_layers"],
        neurons=best_params["neurons"],
        learning_rate=best_params["learning_rate"],
        optimizer_name="adamW",
        dropout=best_params["dropout"],
        weight_decay=best_params["weight_decay"],
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
        [X1_tv, X2_tv],
        y_tv,
        validation_split=0.1,  # small internal val for early stopping only
        epochs=max_epochs,
        batch_size=best_params["batch_size"],
        verbose=1,
        callbacks=[es, ton],
    )

    # Test evaluation
    y_pred = model.predict([X1_test, X2_test], verbose=0).ravel()

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
