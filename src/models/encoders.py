import os

from tensorflow.keras import Model

from models.single_view import build


def load_pretrained_encoder(
    input_dim: int, weights_path: str, params: dict, trainable: bool = False
) -> Model:
    """
    Build the single-view model, load weights, then 'clip off' the regression head
    so that the model outputs the last hidden representation.
    """
    base = build(
        input_dim=input_dim,
        hidden_layers=params["n_layers"],
        neurons=params["neurons"],
        learning_rate=params["learning_rate"],
        optimizer_name="adamW",
        dropout=params["dropout"],
        weight_decay=params["weight_decay"],
    )

    if not os.path.exists(weights_path):
        raise FileNotFoundError(weights_path)

    base.load_weights(weights_path)

    last_hidden_name = f"{base.name}_hidden_{params['n_layers']}"
    encoder_output = base.get_layer(last_hidden_name).output

    encoder = Model(
        inputs=base.input,
        outputs=encoder_output,
        name=os.path.basename(weights_path) + "_encoder",
    )
    encoder.trainable = trainable
    return encoder


def load_single_view_model(
    input_dim: int,
    weights_path: str,
    params: dict,
) -> Model:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(weights_path)

    base = build(
        input_dim=input_dim,
        hidden_layers=params["n_layers"],
        neurons=params["neurons"],
        learning_rate=params["learning_rate"],
        optimizer_name="adamW",
        dropout=params["dropout"],
        weight_decay=params["weight_decay"],
    )
    base.load_weights(weights_path)
    return base
