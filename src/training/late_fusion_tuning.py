import optuna

from training.cross_validation import late_fusion_cv_rmse


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
):
    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        mean_rmse = late_fusion_cv_rmse(
            X1_scaled=X1_scaled,
            X2_scaled=X2_scaled,
            X1_params=X1_params,
            X2_params=X2_params,
            X1_weights_path=X1_weights_path,
            X2_weights_path=X2_weights_path,
            y=y,
            trainval_idx=trainval_idx,
            alpha=alpha,
            n_splits=n_splits,
            max_epochs=max_epochs,
        )
        trial.set_user_attr("alpha", alpha)
        return mean_rmse

    return objective
