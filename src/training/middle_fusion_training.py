from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
)
from scipy.stats import pearsonr
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

from models import encoders, middle_fusion, middle_fusion_attention
from utils import train_mode as mode


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
    train_mode: mode.EncoderTrainMode,
    max_epochs=300,
):
    # Split data
    X1_tv = X1[trainval_idx]
    X2_tv = X2[trainval_idx]
    y_tv = y[trainval_idx]
    X1_test = X1[test_idx]
    X2_test = X2[test_idx]
    y_test = y[test_idx]

    if train_mode == mode.EncoderTrainMode.TRAINED:
        # MF0: random initialisation
        X1_encoder = encoders.build(
            input_dim=X1.shape[1],
            hidden_layers=X1_params["n_layers"],
            neurons=X1_params["neurons"],
            learning_rate=X1_params["learning_rate"],
            optimizer_name="adamW",
            dropout=X1_params["dropout"],
            weight_decay=X1_params["weight_decay"],
            name="expr_encoder",
        )
        X2_encoder = encoders.build(
            input_dim=X2.shape[1],
            hidden_layers=X2_params["n_layers"],
            neurons=X2_params["neurons"],
            learning_rate=X2_params["learning_rate"],
            optimizer_name="adamW",
            dropout=X2_params["dropout"],
            weight_decay=X2_params["weight_decay"],
            name="prot_encoder",
        )

    elif (
        train_mode == mode.EncoderTrainMode.FROZEN
        or train_mode == mode.EncoderTrainMode.ATTENTION
    ):
        # MF1: pretrained but frozen
        X1_encoder = encoders.load_pretrained_encoder(
            input_dim=X1.shape[1],
            weights_path=X1_weights_path,
            params=X1_params,
            trainable=False,
        )
        X2_encoder = encoders.load_pretrained_encoder(
            input_dim=X2.shape[1],
            weights_path=X2_weights_path,
            params=X2_params,
            trainable=False,
        )

    elif train_mode == mode.EncoderTrainMode.FINETUNE:
        # MF2: pretrained and fine-tuned
        X1_encoder = encoders.load_pretrained_encoder(
            input_dim=X1.shape[1],
            weights_path=X1_weights_path,
            params=X1_params,
            trainable=True,
        )
        X2_encoder = encoders.load_pretrained_encoder(
            input_dim=X2.shape[1],
            weights_path=X2_weights_path,
            params=X2_params,
            trainable=True,
        )
    else:
        raise ValueError(f"Unsupported train_mode: {train_mode}")

    if train_mode == mode.EncoderTrainMode.ATTENTION:
        model = middle_fusion_attention.build(
            input_1_dim=X1_tv.shape[1],
            input_2_dim=X2_tv.shape[1],
            encoder_1=X1_encoder,
            encoder_2=X2_encoder,
            hidden_layers=best_params["n_layers"],
            neurons=best_params["neurons"],
            learning_rate=best_params["learning_rate"],
            optimizer_name="adamW",
            dropout=best_params["dropout"],
            weight_decay=best_params["weight_decay"],
            gate_hidden_dim=None,
        )
    else:
        model = middle_fusion.build(
            input_1_dim=X1_tv.shape[1],
            input_2_dim=X2_tv.shape[1],
            encoder_1=X1_encoder,
            encoder_2=X2_encoder,
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
