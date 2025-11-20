import pandas as pd
import json
import os

# All paths defined in ONE place
DATA_DIR = "data/processed"

X_EXPR_FILE = "X_expr.csv"
X_PROT_FILE = "X_prot.csv"
X_FLUX_FILE = "X_flux.csv"
Y_FILE = "y_duibhir.csv"  # growth rates measured in SC medium - used in Culley’s MMANN
# Y_FILE = "y_messner.csv" # growth rates measured in SM medium - from Messner et al.

TEST_IDX_FILE = "data/splits/test_indices.csv"
TRAIN_IDX_FILE = "data/splits/trainval_indices.csv"

BEST_PARAMS_FILE = "results/best_params/params.json"


def load_expression():
    path = os.path.join(DATA_DIR, X_EXPR_FILE)
    return pd.read_csv(path, index_col=0).values


def load_proteomics():
    path = os.path.join(DATA_DIR, X_PROT_FILE)
    return pd.read_csv(path, index_col=0).values


def load_flux():
    path = os.path.join(DATA_DIR, X_FLUX_FILE)
    return pd.read_csv(path, index_col=0).values


def load_target():
    path = os.path.join(DATA_DIR, Y_FILE)
    return pd.read_csv(path, index_col=0)["growth"].values


def load_test_indices():
    return pd.read_csv(TEST_IDX_FILE, header=None).iloc[:, 0].values.astype(int)


def load_train_indices():
    return pd.read_csv(TRAIN_IDX_FILE, header=None).iloc[:, 0].values.astype(int)


def get_loader_for_modality(modality: str):
    modality = modality.lower()
    if modality in ("expr", "expression"):
        return load_expression
    elif modality in ("prot", "proteomics"):
        return load_proteomics
    elif modality == "flux":
        return load_flux
    else:
        raise ValueError(f"Unknown modality: {modality}")


def load_best_params(model_id: str) -> dict:
    with open(BEST_PARAMS_FILE, "r") as f:
        all_params = json.load(f)
    if model_id not in all_params:
        raise ValueError(f"Model ID '{model_id}' not found in best params file.")
    return all_params[model_id]
