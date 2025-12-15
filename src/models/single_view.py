import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout


def build(
    input_dim: int,
    hidden_layers: int,
    neurons: int,
    learning_rate: float,
    optimizer_name: str,
    dropout: float,
    weight_decay: float,
    name: str = "single_view_mlp",
):

    # Use the model name as a prefix for layer names to avoid clashes
    inputs = Input(shape=(input_dim,), name=f"{name}_input")
    x = inputs

    for i in range(hidden_layers):
        x = Dense(
            neurons,
            activation="relu",
            name=f"{name}_hidden_{i+1}",
        )(x)
        x = Dropout(dropout, name=f"{name}_dropout_{i+1}")(x)

    outputs = Dense(1, activation="linear", name=f"{name}_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name=name)

    if optimizer_name.lower() == "adamw":
        opt = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimiser: {optimizer_name}")

    model.compile(
        loss="mean_squared_error",
        optimizer=opt,
        metrics=["mean_absolute_error"],
    )
    return model
