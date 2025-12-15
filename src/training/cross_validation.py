from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

from models import single_view, middle_fusion, middle_fusion_attention, encoders

from utils import train_mode as mode


from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras import backend as K
import gc


def cv_rmse(
    X,
    y,
    trainval_idx,
    config: dict,
    n_splits: int = 5,
    max_epochs: int = 300,
    random_state: int = 123,
):
    """
    X, y: full dataset (numpy arrays or pandas .values)
    trainval_idx: indices used for hyperparameter tuning (no test leakage)
    config: single hyperparameter configuration (from Optuna)

    Returns
    -------
    mean_rmse : float
        Mean RMSE across folds for this config.
    """

    # Use only train/val indices
    X_tv = X[trainval_idx]
    y_tv = y[trainval_idx]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_rmses = []

    for fold, (train_idx_local, val_idx_local) in enumerate(kf.split(X_tv), start=1):
        X_tr = X_tv[train_idx_local]
        y_tr = y_tv[train_idx_local]
        X_val = X_tv[val_idx_local]
        y_val = y_tv[val_idx_local]

        # Build model – adjust argument names to match your single_view.build
        model = single_view.build(
            input_dim=X_tv.shape[1],
            hidden_layers=config["n_layers"],
            neurons=config["neurons"],
            learning_rate=config["learning_rate"],
            dropout=config["dropout"],
            weight_decay=config["weight_decay"],
            optimizer_name="adamW",
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
            batch_size=config["batch_size"],
            verbose=0,
            callbacks=[es, ton],
        )

        y_val_pred = model.predict(X_val, verbose=0).ravel()

        # Guard against non-finite predictions – heavily penalise if it happens
        if not np.all(np.isfinite(y_val_pred)):
            rmse = 1e9
        else:
            rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))

        fold_rmses.append(rmse)

        # Free TF/Keras resources for this model
        K.clear_session()
        gc.collect()

    mean_rmse = float(np.mean(fold_rmses))
    return mean_rmse


def middle_fusion_cv_rmse(
    X1_scaled,
    X2_scaled,
    X1_weights_path,
    X2_weights_path,
    X1_params,
    X2_params,
    y,
    trainval_idx,
    config: dict,
    train_mode: mode.EncoderTrainMode,
    n_splits: int = 5,
    max_epochs: int = 300,
    random_state: int = 123,
    # trainable: bool = False,
):
    """
    X, y: full dataset (numpy arrays or pandas .values)
    trainval_idx: indices used for hyperparameter tuning (no test leakage)
    config: single hyperparameter configuration (from Optuna)

    Returns
    -------
    mean_rmse : float
        Mean RMSE across folds for this config.
    """

    # Use only train/val indices
    X1_tv = X1_scaled[trainval_idx]
    X2_tv = X2_scaled[trainval_idx]
    y_tv = y[trainval_idx]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_rmses = []

    for fold, (train_idx_local, val_idx_local) in enumerate(kf.split(X1_tv), start=1):
        X1_tr = X1_tv[train_idx_local]
        X2_tr = X2_tv[train_idx_local]
        y_tr = y_tv[train_idx_local]

        X1_val = X1_tv[val_idx_local]
        X2_val = X2_tv[val_idx_local]
        y_val = y_tv[val_idx_local]

        if train_mode == mode.EncoderTrainMode.TRAINED:
            # MF0: random initialisation
            X1_encoder = encoders.build(
                input_dim=X1_scaled.shape[1],
                hidden_layers=X1_params["n_layers"],
                neurons=X1_params["neurons"],
                learning_rate=X1_params["learning_rate"],
                optimizer_name="adamW",
                dropout=X1_params["dropout"],
                weight_decay=X1_params["weight_decay"],
                name="expr_encoder",
            )
            X2_encoder = encoders.build(
                input_dim=X2_scaled.shape[1],
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
                input_dim=X1_scaled.shape[1],
                weights_path=X1_weights_path,
                params=X1_params,
                trainable=False,
            )
            X2_encoder = encoders.load_pretrained_encoder(
                input_dim=X2_scaled.shape[1],
                weights_path=X2_weights_path,
                params=X2_params,
                trainable=False,
            )

        elif train_mode == mode.EncoderTrainMode.FINETUNE:
            # MF2: pretrained and fine-tuned
            X1_encoder = encoders.load_pretrained_encoder(
                input_dim=X1_scaled.shape[1],
                weights_path=X1_weights_path,
                params=X1_params,
                trainable=True,
            )
            X2_encoder = encoders.load_pretrained_encoder(
                input_dim=X2_scaled.shape[1],
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
                hidden_layers=config["n_layers"],
                neurons=config["neurons"],
                learning_rate=config["learning_rate"],
                optimizer_name="adamW",
                dropout=config["dropout"],
                weight_decay=config["weight_decay"],
                gate_hidden_dim=None,  # or e.g. 64 if you want a non-trivial gate MLP
            )
        else:
            model = middle_fusion.build(
                input_1_dim=X1_tv.shape[1],
                input_2_dim=X2_tv.shape[1],
                encoder_1=X1_encoder,
                encoder_2=X2_encoder,
                hidden_layers=config["n_layers"],
                neurons=config["neurons"],
                learning_rate=config["learning_rate"],
                optimizer_name="adamW",
                dropout=config["dropout"],
                weight_decay=config["weight_decay"],
            )

        es = EarlyStopping(
            monitor="val_loss",
            patience=25,
            restore_best_weights=True,
            verbose=0,
        )
        ton = TerminateOnNaN()

        model.fit(
            [X1_tr, X2_tr],
            y_tr,
            validation_data=([X1_val, X2_val], y_val),
            epochs=max_epochs,
            batch_size=config["batch_size"],
            verbose=0,
            callbacks=[es, ton],
        )

        y_val_pred = model.predict([X1_val, X2_val], verbose=0).ravel()

        # Guard against non-finite predictions – heavily penalise if it happens
        if not np.all(np.isfinite(y_val_pred)):
            rmse = 1e9
        else:
            rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))

        fold_rmses.append(rmse)

        # Free TF/Keras resources for this model
        K.clear_session()
        gc.collect()

    mean_rmse = float(np.mean(fold_rmses))
    return mean_rmse


def late_fusion_cv_rmse(
    X1_scaled,
    X2_scaled,
    X1_weights_path,
    X2_weights_path,
    X1_params,
    X2_params,
    y,
    trainval_idx,
    alpha: float,
    n_splits: int = 5,
    max_epochs: int = 300,
    random_state: int = 123,
):
    X1_tv = X1_scaled[trainval_idx]
    X2_tv = X2_scaled[trainval_idx]
    y_tv = y[trainval_idx]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_rmses = []

    for fold, (train_idx_local, val_idx_local) in enumerate(kf.split(X1_tv), start=1):
        X1_tr = X1_tv[train_idx_local]
        X2_tr = X2_tv[train_idx_local]
        y_tr = y_tv[train_idx_local]

        X1_val = X1_tv[val_idx_local]
        X2_val = X2_tv[val_idx_local]
        y_val = y_tv[val_idx_local]

        # Build fresh single-view models for each fold
        model_1 = encoders.load_single_view_model(
            input_dim=X1_tv.shape[1],
            weights_path=X1_weights_path,
            params=X1_params,
            trainable=False,
        )
        # Build fresh single-view models for each fold
        model_2 = encoders.load_single_view_model(
            input_dim=X2_tv.shape[1],
            weights_path=X2_weights_path,
            params=X2_params,
            trainable=False,
        )

        es = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
        ton = TerminateOnNaN()

        # Train single-view models on their own inputs
        model_1.fit(
            X1_tr,
            y_tr,
            validation_data=(X1_val, y_val),
            epochs=max_epochs,
            batch_size=X1_params["batch_size"],
            verbose=0,
            callbacks=[es, ton],
        )

        model_2.fit(
            X2_tr,
            y_tr,
            validation_data=(X2_val, y_val),
            epochs=max_epochs,
            batch_size=X2_params["batch_size"],
            verbose=0,
            callbacks=[es, ton],
        )

        # Predictions on validation folds
        y_val_pred_1 = model_1.predict(X1_val, verbose=0).ravel()
        y_val_pred_2 = model_2.predict(X2_val, verbose=0).ravel()

        # Late fusion of predictions
        y_val_pred_late = alpha * y_val_pred_1 + (1.0 - alpha) * y_val_pred_2

        if not np.all(np.isfinite(y_val_pred_late)):
            rmse = 1e9
        else:
            rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred_late)))

        fold_rmses.append(rmse)

        K.clear_session()
        gc.collect()

    return float(np.mean(fold_rmses))
