# models/single_view.py
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.constraints import max_norm


def build(
    input_dim: int,
    hidden_layers: int,
    neurons: int,
    learning_rate: float,
    optimizer_name: str,
    dropout: float,
    kernel_constraint,
):
    # kernel constraint
    constraint = max_norm(kernel_constraint) if kernel_constraint is not None else None

    inputs = Input(shape=(input_dim,), name="input")
    x = inputs

    for i in range(hidden_layers):
        x = Dense(
            neurons,
            activation="relu",
            kernel_constraint=constraint,
            name=f"hidden_{i+1}",
        )(x)
        x = Dropout(dropout, name=f"dropout_{i+1}")(x)

    outputs = Dense(1, activation="linear", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="single_view_mlp")

    if optimizer_name.lower() == "adam":
        opt = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
        )
    else:  # "sgd"
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    model.compile(
        loss="mean_squared_error",
        optimizer=opt,
        metrics=["mean_absolute_error"],
    )
    return model
