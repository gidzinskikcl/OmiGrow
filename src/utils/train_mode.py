import enum


class EncoderTrainMode(enum.Enum):
    TRAINED = "trained"
    FROZEN = "frozen"
    FINETUNE = "finetune"
    ATTENTION = "attention"


def get_train_mode(model_id: str) -> EncoderTrainMode:
    if model_id == "MF0":
        result = EncoderTrainMode.TRAINED
    elif model_id == "MF1":
        result = EncoderTrainMode.FROZEN
    elif model_id == "MF2":
        result = EncoderTrainMode.FINETUNE
    elif model_id == "MF3":
        result = EncoderTrainMode.ATTENTION
    else:
        raise ValueError(f"Unsupported model_id: {model_id}")
    return result
