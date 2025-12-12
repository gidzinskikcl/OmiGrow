import optuna

from training.cross_validation import middle_fusion_cv_rmse
from training.config import build_config_from_trial


def evaluate_config(
    X1_scaled,
    X2_scaled,
    X1_weights_path,
    X2_weights_path,
    X1_params,
    X2_params,
    y,
    trainval_idx,
    config: dict,
    n_splits: int = 5,
    max_epochs: int = 300,
    trainable: bool = False,
):
    """
    Run CV for a single hyperparameter config and return mean RMSE (or your chosen metric).
    """
    mean_rmse = middle_fusion_cv_rmse(
        X1_scaled=X1_scaled,
        X2_scaled=X2_scaled,
        X1_weights_path=X1_weights_path,
        X2_weights_path=X2_weights_path,
        X1_params=X1_params,
        X2_params=X2_params,
        y=y,
        trainval_idx=trainval_idx,
        config=config,
        n_splits=n_splits,
        max_epochs=max_epochs,
        trainable=trainable,
    )
    return mean_rmse


def make_objective(
    X1_scaled,
    X2_scaled,
    X1_weights_path,
    X2_weights_path,
    X1_params,
    X2_params,
    y,
    trainval_idx,
    n_splits,
    max_epochs,
    trainable: bool = False,
):
    def objective(trial: optuna.Trial) -> float:
        config = build_config_from_trial(trial)
        mean_rmse = evaluate_config(
            X1_scaled=X1_scaled,
            X2_scaled=X2_scaled,
            X1_weights_path=X1_weights_path,
            X2_weights_path=X2_weights_path,
            X1_params=X1_params,
            X2_params=X2_params,
            y=y,
            trainval_idx=trainval_idx,
            config=config,
            n_splits=n_splits,
            max_epochs=max_epochs,
            trainable=trainable,
        )
        trial.set_user_attr("config", config)
        return mean_rmse

    return objective
