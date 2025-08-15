import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs

"""Minimal XTTS embedding script with safe global registration."""

# Register the configuration classes so XTTS weights can be loaded with
# ``torch.load(weights_only=True)``. This mirrors the approach used in
# ``xtts_match_emidunno.py`` lines 90-103.
torch.serialization.add_safe_globals(
    [XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]
)


def load_xtts_model(
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
) -> TTS:
    """Load the XTTS model using the provided model name."""

    return TTS(model_name)


if __name__ == "__main__":
    tts = load_xtts_model()
    print("XTTS model loaded successfully.")
