from TTS.api import TTS
import os
import numpy as np
import torch

# Model path
MODEL_DIR = r"C:\xtts_model_v2"

# Create TTS object
tts = TTS()  # Don't pass anything yet
tts.load_model(
    model_path=os.path.join(MODEL_DIR, "model.pth"),
    config_path=os.path.join(MODEL_DIR, "config.json"),
    speakers_file=os.path.join(MODEL_DIR, "speakers_xtts.pth"),
    vocab_path=os.path.join(MODEL_DIR, "vocab.json"),
    use_cuda=False
)

# Paths
WAV_DIR = r"C:\Users\Lenovo\Desktop\500wav_french"
OUT_DIR = r"C:\Users\Lenovo\Desktop\xtts_french_embed"
os.makedirs(OUT_DIR, exist_ok=True)

# Extract embeddings
def extract_embeddings(wav_dir, out_dir):
    for file in os.listdir(wav_dir):
        if file.endswith(".wav"):
            filepath = os.path.join(wav_dir, file)
            try:
                speaker_embed, _, _ = tts.get_conditioning_latents(audio_path=filepath)
                np.save(os.path.join(out_dir, file.replace(".wav", ".npy")), speaker_embed.cpu().numpy())
                print(f"✔ Saved: {file}")
            except Exception as e:
                print(f"✖ Failed: {file} — {e}")

extract_embeddings(WAV_DIR, OUT_DIR)
print(f"All embeddings saved to: {OUT_DIR}")

