def build_config_from_trial(trial):

    return {
        # ----------------------------------------------------
        # Architecture
        # ----------------------------------------------------
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        # Neurons per layer (choose one size and repeat it n_layers times)
        "neurons": trial.suggest_categorical("neurons", [64, 128, 256]),
        # ----------------------------------------------------
        # Optimisation
        # ----------------------------------------------------
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-4,
            1e-2,
            log=True,
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size",
            [32, 64, 128],
        ),
        # ----------------------------------------------------
        # Regularisation
        # ----------------------------------------------------
        "dropout": trial.suggest_float(
            "dropout",
            0.0,
            0.5,
        ),
        "weight_decay": trial.suggest_categorical(
            "weight_decay",
            [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
        ),
    }
