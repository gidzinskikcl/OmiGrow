import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Concatenate,
    Lambda,
    Add,
    Activation,
)
from tensorflow.keras.regularizers import l2


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
    weight_decay: float,
    gate_hidden_dim: int | None = None,
) -> Model:
    """
    Attention-based middle fusion:
      - encoder_1, encoder_2: modality-specific encoders (expr, prot)
      - gating network computes softmax weights for each modality per sample
      - fused representation = g1 * h1 + g2 * h2
      - fusion head (MLP) on fused representation predicts growth
    """

    # ----------------------------------------------------------------------
    # Inputs
    # ----------------------------------------------------------------------
    inp1 = Input(shape=(input_1_dim,), name="input_1")
    inp2 = Input(shape=(input_2_dim,), name="input_2")

    # ----------------------------------------------------------------------
    # Encoders
    # encoder_1, encoder_2 should output latent representations, not scalars
    # ----------------------------------------------------------------------
    h1 = encoder_1(inp1)  # shape: (batch, d_h1)
    h2 = encoder_2(inp2)  # shape: (batch, d_h2)

    # ----------------------------------------------------------------------
    # Gating network (attention over modalities)
    # ----------------------------------------------------------------------
    # Concatenate the two representations
    gate_input = Concatenate(name="gate_concat")([h1, h2])

    x_gate = gate_input
    if gate_hidden_dim is not None and gate_hidden_dim > 0:
        x_gate = Dense(
            gate_hidden_dim,
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            name="gate_hidden",
        )(x_gate)

    gate_logits = Dense(
        2,
        kernel_regularizer=l2(weight_decay),
        name="gate_logits",
    )(x_gate)
    gate = Activation("softmax", name="gate_softmax")(gate_logits)  # (batch, 2)

    # Split into g1, g2 with shape (batch, 1)
    g1 = Lambda(lambda g: g[:, 0:1], name="gate_weight_1")(gate)
    g2 = Lambda(lambda g: g[:, 1:2], name="gate_weight_2")(gate)

    # Weighted representations
    h1_w = Lambda(lambda inputs: inputs[0] * inputs[1], name="weighted_h1")([g1, h1])
    h2_w = Lambda(lambda inputs: inputs[0] * inputs[1], name="weighted_h2")([g2, h2])

    # Fused representation
    h_fused = Add(name="fused_representation")([h1_w, h2_w])

    # ----------------------------------------------------------------------
    # Fusion head (same style as your current middle_fusion)
    # ----------------------------------------------------------------------
    x = h_fused
    for i in range(hidden_layers):
        x = Dense(
            neurons,
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            name=f"fusion_hidden_{i+1}",
        )(x)
        x = Dropout(dropout, name=f"fusion_dropout_{i+1}")(x)

    outputs = Dense(1, activation="linear", name="output")(x)

    model = Model(
        inputs=[inp1, inp2],
        outputs=outputs,
        name="middle_fusion_attention",
    )

    # ----------------------------------------------------------------------
    # Optimizer and compile
    # ----------------------------------------------------------------------
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
