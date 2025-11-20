from sklearn.model_selection import ParameterGrid

param_grid = {
    "hidden_layers": [1, 2],
    "neurons": [50, 100, 200, 400],
    "optimizer": ["adam", "sgd"],
    "learning_rate": [0.0005, 0.001, 0.005, 0.01, 0.05],
    "dropout": [0.2, 0.4],
    "kernel_constraint": [None, 3],
}

grid = list(ParameterGrid(param_grid))
