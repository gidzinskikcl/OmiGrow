import optuna

from training.cross_validation import cv_rmse
from training.config import build_config_from_trial


def evaluate_config(
    X,
    y,
    trainval_idx,
    config: dict,
    n_splits: int = 5,
    max_epochs: int = 300,
):
    """
    Run CV for a single hyperparameter config and return mean RMSE (or your chosen metric).
    """
    mean_rmse = cv_rmse(
        X=X,
        y=y,
        trainval_idx=trainval_idx,
        config=config,
        n_splits=n_splits,
        max_epochs=max_epochs,
    )
    return mean_rmse


def make_objective(X_scaled, y, trainval_idx, n_splits, max_epochs):
    def objective(trial: optuna.Trial) -> float:
        config = build_config_from_trial(trial)
        mean_rmse = evaluate_config(
            X=X_scaled,
            y=y,
            trainval_idx=trainval_idx,
            config=config,
            n_splits=n_splits,
            max_epochs=max_epochs,
        )
        trial.set_user_attr("config", config)
        return mean_rmse

    return objective
