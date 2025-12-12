import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Concatenate


def build(
    input_1_dim: int,
    input_2_dim: int,
    encoder_1: Model,
    encoder_2: Model,
    hidden_layers: int,
    neurons: int,
    learning_rate: float,
    optimizer_name: str,
    dropout: float,
    weight_decay,
):
    """
    Middle fusion:
      X1 -> encoder_1 -> z1
      X2 -> encoder_2 -> z2
      [z1, z2] -> Dense(fusion_neurons) -> Dense(1)
    """

    inp1 = Input(shape=(input_1_dim,), name="input_mod1")
    inp2 = Input(shape=(input_2_dim,), name="input_mod2")

    z1 = encoder_1(inp1)
    z2 = encoder_2(inp2)

    x = Concatenate(name="fusion_concat")([z1, z2])

    for i in range(hidden_layers):
        x = Dense(
            neurons,
            activation="relu",
            name=f"hidden_{i+1}",
        )(x)
        x = Dropout(dropout, name=f"dropout_{i+1}")(x)

    outputs = Dense(1, activation="linear", name="output")(x)

    model = Model(inputs=[inp1, inp2], outputs=outputs, name="middle_fusion")

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
