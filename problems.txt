(xtts-env) C:\Windows\System32>python "C:\Users\Lenovo\.spyder-py3\xtts_match_embed.py"
===== DEPENDENCY CHECK =====
TTS already installed
ffmpeg-python already installed
librosa already installed
scipy already installed
numpy already installed
pandas already installed
torch already installed
19:49:07 [INFO] ===== PHONETIC EMBEDDING EXTRACTION STARTED =====
19:49:07 [INFO]
=== SYSTEM INFORMATION ===
Python: 3.10.16
Platform: win32
Working Dir: C:\Windows\System32

=== PACKAGE VERSIONS ===
PyTorch: 2.7.0+cpu
CUDA Available: False
TTS: 0.22.0
Librosa: 0.10.0
19:49:07 [INFO] Initializing XTTS model...
19:49:07 [INFO]  > tts_models/multilingual/multi-dataset/xtts_v2 is already downloaded.
19:49:11 [INFO]  > Using model: xtts
19:49:53 [ERROR] Model load failed: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
        (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
        (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
        WeightsUnpickler error: Unsupported global: GLOBAL TTS.config.shared_configs.BaseDatasetConfig was not an allowed global by default. Please use `torch.serialization.add_safe_globals([TTS.config.shared_configs.BaseDatasetConfig])` or the `torch.serialization.safe_globals([TTS.config.shared_configs.BaseDatasetConfig])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
19:49:53 [ERROR] Failed to load XTTS model. Exiting.

(xtts-env) C:\Windows\System32>python import os
python: can't open file 'C:\\Windows\\System32\\import': [Errno 2] No such file or directory

(xtts-env) C:\Windows\System32>import sys
'import' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>import subprocess
'import' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>import logging
'import' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>import glob
'import' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>import importlib.util
'import' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32># Explicit dependency mapping (PyPI package -> import name)
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>DEPENDENCY_MAP = {
'DEPENDENCY_MAP' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    "TTS": "TTS",
'"TTS":' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    "librosa": "librosa",
'"librosa":' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    "scipy": "scipy",
'"scipy":' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    "numpy": "numpy",
'"numpy":' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    "pandas": "pandas",
'"pandas":' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    "torch": "torch",
'"torch":' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    "soundfile": "soundfile"  # Replaces librosa for faster loading
'"soundfile":' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>}
'}' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32># Dependency check must come FIRST
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>def check_dependencies():
'def' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    """Install missing dependencies with explicit mapping"""
'"""Install missing dependencies with explicit mapping"""' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    print("===== DEPENDENCY CHECK =====")
Unable to initialize device PRN

(xtts-env) C:\Windows\System32>    installed_anything = False
'installed_anything' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    for pkg, import_name in DEPENDENCY_MAP.items():
pkg was unexpected at this time.
(xtts-env) C:\Windows\System32>        # Check using importlib to handle dotted paths
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        spec = importlib.util.find_spec(import_name.split('.')[0])
'spec' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        if spec is None:
is was unexpected at this time.

(xtts-env) C:\Windows\System32>            print(f"Installing {pkg}...")
Unable to initialize device PRN

(xtts-env) C:\Windows\System32>            try:
'try:' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>                subprocess.check_call(
'subprocess.check_call' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>                    [sys.executable, "-m", "pip", "install", pkg],
'[sys.executable' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>                    stdout=subprocess.DEVNULL,
'stdout' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>                    stderr=subprocess.DEVNULL
'stderr' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>                )
(xtts-env) C:\Windows\System32>                print(f"Installed {pkg}")
Unable to initialize device PRN

(xtts-env) C:\Windows\System32>                installed_anything = True
'installed_anything' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>            except Exception as e:
'except' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>                print(f"Failed to install {pkg}: {str(e)}", file=sys.stderr)
Unable to initialize device PRN

(xtts-env) C:\Windows\System32>                sys.exit(1)
'sys.exit' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        else:
'else:' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>            print(f"{pkg} already installed")
Unable to initialize device PRN

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    if installed_anything:
The syntax of the command is incorrect.
(xtts-env) C:\Windows\System32>        print("\n!!! RESTART REQUIRED !!! New dependencies installed.", file=sys.stderr)
Unable to initialize device PRN

(xtts-env) C:\Windows\System32>        sys.exit(86)  # Special exit code for restart
'sys.exit' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32># Check dependencies before any other imports
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>check_dependencies()
'check_dependencies' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32># Now safe to import everything
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>import numpy as np
'import' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>import pandas as pd
'import' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>import soundfile as sf  # Faster than librosa
'import' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>from scipy.spatial.distance import cosine
'from' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>import torch
'import' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>from TTS.api import TTS
'from' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
'from' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32># Initialize torch serialization safely
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig])
'torch.serialization.add_safe_globals' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32># Enhanced logging configuration
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>class StreamToLogger:
'class' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    def __init__(self, logger, log_level=logging.INFO):
'def' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        self.logger = logger
'self.logger' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        self.log_level = log_level
'self.log_level' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    def write(self, buf):
'def' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        for line in buf.rstrip().splitlines():
line was unexpected at this time.

(xtts-env) C:\Windows\System32>            if line.strip():
The syntax of the command is incorrect.

(xtts-env) C:\Windows\System32>                self.logger.log(self.log_level, line.rstrip())
'self.logger.log' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    def flush(self):
'def' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        # Robust flush handling
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        for handler in self.logger.handlers:
handler was unexpected at this time.

(xtts-env) C:\Windows\System32>            flush_func = getattr(handler, 'flush', None)
'flush_func' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>            if callable(flush_func):
The syntax of the command is incorrect.

(xtts-env) C:\Windows\System32>                flush_func()
'flush_func' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32># Configure dual logging
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>logging.basicConfig(
'logging.basicConfig' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    level=logging.INFO,
'level' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    format="%(asctime)s [%(levelname)s] %(message)s",
Invalid drive specification.

(xtts-env) C:\Windows\System32>    datefmt="%H:%M:%S",
'datefmt' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    handlers=[
'handlers' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        logging.FileHandler("xtts_progress.log"),
'logging.FileHandler' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        logging.StreamHandler()
'logging.StreamHandler' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    ]
']' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>)
(xtts-env) C:\Windows\System32>logger = logging.getLogger('XTTS')
'logger' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>sys.stdout = StreamToLogger(logger, logging.INFO)
'sys.stdout' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>sys.stderr = StreamToLogger(logger, logging.ERROR)
'sys.stderr' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>def print_system_info():
'def' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    """Display system and package info"""
'"""Display system and package info"""' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    info = [
'info' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        "\n=== SYSTEM INFORMATION ===",
'"\n=== SYSTEM INFORMATION ==="' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        f"Python: {sys.version.split()[0]}",
The filename, directory name, or volume label syntax is incorrect.

(xtts-env) C:\Windows\System32>        f"Platform: {sys.platform}",
The filename, directory name, or volume label syntax is incorrect.

(xtts-env) C:\Windows\System32>        f"Working Dir: {os.getcwd()}",
The filename, directory name, or volume label syntax is incorrect.

(xtts-env) C:\Windows\System32>        "\n=== PACKAGE VERSIONS ==="
'"\n=== PACKAGE VERSIONS ==="' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    ]
']' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    for pkg in DEPENDENCY_MAP.values():
pkg was unexpected at this time.
(xtts-env) C:\Windows\System32>        try:
'try:' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>            mod = __import__(pkg)
'mod' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>            info.append(f"{pkg}: {getattr(mod, '__version__', 'unknown')}")
'info.append' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        except ImportError:
'except' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>            info.append(f"{pkg}: not available")
'info.append' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    logger.info("\n".join(info))
'logger.info' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>def load_xtts_model(force_cpu=False):
'def' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    """Load model with secure weights handling"""
'"""Load model with secure weights handling"""' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    try:
'try:' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        logger.info("Initializing XTTS model...")
'logger.info' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        device = "cpu" if force_cpu else "cuda" if torch.cuda.is_available() else "cpu"
'device' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        tts = TTS(
usage: tts [-h] [--list_models [LIST_MODELS]] [--model_info_by_idx MODEL_INFO_BY_IDX] [--model_info_by_name MODEL_INFO_BY_NAME] [--text TEXT] [--model_name MODEL_NAME] [--vocoder_name VOCODER_NAME]
           [--config_path CONFIG_PATH] [--model_path MODEL_PATH] [--out_path OUT_PATH] [--use_cuda USE_CUDA] [--device DEVICE] [--vocoder_path VOCODER_PATH] [--vocoder_config_path VOCODER_CONFIG_PATH]
           [--encoder_path ENCODER_PATH] [--encoder_config_path ENCODER_CONFIG_PATH] [--pipe_out [PIPE_OUT]] [--speakers_file_path SPEAKERS_FILE_PATH] [--language_ids_file_path LANGUAGE_IDS_FILE_PATH]
           [--speaker_idx SPEAKER_IDX] [--language_idx LANGUAGE_IDX] [--speaker_wav SPEAKER_WAV [SPEAKER_WAV ...]] [--gst_style GST_STYLE] [--capacitron_style_wav CAPACITRON_STYLE_WAV]
           [--capacitron_style_text CAPACITRON_STYLE_TEXT] [--list_speaker_idxs [LIST_SPEAKER_IDXS]] [--list_language_idxs [LIST_LANGUAGE_IDXS]] [--save_spectogram SAVE_SPECTOGRAM]
           [--reference_wav REFERENCE_WAV] [--reference_speaker_idx REFERENCE_SPEAKER_IDX] [--progress_bar PROGRESS_BAR] [--source_wav SOURCE_WAV] [--target_wav TARGET_WAV] [--voice_dir VOICE_DIR]
tts: error: unrecognized arguments: = TTS(

(xtts-env) C:\Windows\System32>            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
'model_name' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>            progress_bar=False,
'progress_bar' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>            gpu=(device == "cuda")
'gpu' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        )
(xtts-env) C:\Windows\System32>        logger.info(f"XTTS loaded on {device.upper()}")
'logger.info' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        return tts
'return' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    except Exception as e:
'except' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        logger.error(f"Model load failed: {str(e)}")
'logger.error' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        return None
'return' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>@torch.inference_mode()
'torch.inference_mode' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>def extract_speaker_embedding(audio_path, tts):
'def' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    """Extract speaker embedding with API version fallback"""
'"""Extract speaker embedding with API version fallback"""' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    try:
'try:' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        logger.info(f"Processing: {os.path.basename(audio_path)}")
'logger.info' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>        # Load audio with soundfile (faster than librosa)
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        audio, sr = sf.read(audio_path)
'audio' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        logger.info(f"Audio loaded: {len(audio)/sr:.2f}s @ {sr}Hz")
'logger.info' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>        # API version fallback
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        try:
'try:' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>            # Try new API (TTS >= 0.22)
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>            speaker_latent, _, _ = tts.tts_model.get_conditioning_latents(audio, sr)
'speaker_latent' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        except AttributeError:
'except' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>            # Fallback to old API (TTS < 0.22)
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>            speaker_latent, _, _ = tts.synthesizer.tts_model.get_conditioning_latents(audio, sr)
'speaker_latent' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>        # Convert to numpy array
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        embedding = speaker_latent.cpu().numpy()
'embedding' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        logger.info(f"Speaker embedding shape: {embedding.shape}")
'logger.info' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        return embedding
'return' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    except Exception as e:
'except' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        logger.error(f"Processing failed: {str(e)}")
'logger.error' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        return None
'return' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>def cosine_similarity(u, v):
'def' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    """Numerically stable cosine similarity (range: -1 to 1)"""
The filename, directory name, or volume label syntax is incorrect.

(xtts-env) C:\Windows\System32>    u_flat = u.flatten().astype(np.float64)
'u_flat' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    v_flat = v.flatten().astype(np.float64)
'v_flat' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    norm_u = np.linalg.norm(u_flat)
'norm_u' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    norm_v = np.linalg.norm(v_flat)
'norm_v' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    # Handle zero vectors
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    if norm_u == 0 or norm_v == 0:

(xtts-env) C:\Windows\System32>        return 0.0
'return' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    return np.dot(u_flat, v_flat) / (norm_u * norm_v)
'return' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>def process_directory(directory, tts, lang_prefix):
'def' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    """Process all WAV files recursively with language prefix"""
'"""Process all WAV files recursively with language prefix"""' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    embeddings = {}
'embeddings' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    wav_files = glob.glob(os.path.join(directory, "**", "*.wav"), recursive=True)
'wav_files' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    if not wav_files:
The syntax of the command is incorrect.
(xtts-env) C:\Windows\System32>        logger.warning(f"No WAV files found in {directory}")
'logger.warning' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        return embeddings
'return' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    for audio_path in wav_files:
audio_path was unexpected at this time.
(xtts-env) C:\Windows\System32>        embedding = extract_speaker_embedding(audio_path, tts)
'embedding' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        if embedding is not None:
is was unexpected at this time.

(xtts-env) C:\Windows\System32>            # Prefix with language to avoid collisions
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>            rel_path = f"{lang_prefix}/{os.path.relpath(audio_path, directory)}"
'rel_path' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>            embeddings[rel_path] = embedding
'embeddings[rel_path]' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    return embeddings
'return' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>def save_embeddings(embeddings, filename="voice_embeddings.npz"):
'def' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    """Save embeddings as float32 arrays"""
'"""Save embeddings as float32 arrays"""' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    np.savez(
'np.savez' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        filename,
'filename' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        **{k: v.astype(np.float32) for k, v in embeddings.items()}
'**{k:' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    )
(xtts-env) C:\Windows\System32>    logger.info(f"Saved embeddings to {filename}")
'logger.info' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>def calculate_similarity(embeddings):
'def' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    """Vectorized cosine similarity calculation"""
'"""Vectorized cosine similarity calculation"""' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    filenames = list(embeddings.keys())
'filenames' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    embeddings_arr = np.array([emb.flatten() for emb in embeddings.values()])
'embeddings_arr' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    # Normalize embeddings
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
'norms' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    norms[norms == 0] = 1  # Prevent division by zero
'norms[norms' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    embeddings_norm = embeddings_arr / norms
'embeddings_norm' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    # Vectorized similarity matrix
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
'similarity_matrix' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    return pd.DataFrame(similarity_matrix, index=filenames, columns=filenames)
'return' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>def main():
'def' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    """Main workflow with error handling"""
'"""Main workflow with error handling"""' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    logger.info("===== PHONETIC EMBEDDING EXTRACTION STARTED =====")
'logger.info' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    print_system_info()
'print_system_info' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    # Load model (optionally force CPU)
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    tts = load_xtts_model(force_cpu=False)
usage: tts [-h] [--list_models [LIST_MODELS]] [--model_info_by_idx MODEL_INFO_BY_IDX] [--model_info_by_name MODEL_INFO_BY_NAME] [--text TEXT] [--model_name MODEL_NAME] [--vocoder_name VOCODER_NAME]
           [--config_path CONFIG_PATH] [--model_path MODEL_PATH] [--out_path OUT_PATH] [--use_cuda USE_CUDA] [--device DEVICE] [--vocoder_path VOCODER_PATH] [--vocoder_config_path VOCODER_CONFIG_PATH]
           [--encoder_path ENCODER_PATH] [--encoder_config_path ENCODER_CONFIG_PATH] [--pipe_out [PIPE_OUT]] [--speakers_file_path SPEAKERS_FILE_PATH] [--language_ids_file_path LANGUAGE_IDS_FILE_PATH]
           [--speaker_idx SPEAKER_IDX] [--language_idx LANGUAGE_IDX] [--speaker_wav SPEAKER_WAV [SPEAKER_WAV ...]] [--gst_style GST_STYLE] [--capacitron_style_wav CAPACITRON_STYLE_WAV]
           [--capacitron_style_text CAPACITRON_STYLE_TEXT] [--list_speaker_idxs [LIST_SPEAKER_IDXS]] [--list_language_idxs [LIST_LANGUAGE_IDXS]] [--save_spectogram SAVE_SPECTOGRAM]
           [--reference_wav REFERENCE_WAV] [--reference_speaker_idx REFERENCE_SPEAKER_IDX] [--progress_bar PROGRESS_BAR] [--source_wav SOURCE_WAV] [--target_wav TARGET_WAV] [--voice_dir VOICE_DIR]
tts: error: unrecognized arguments: = load_xtts_model(force_cpu=False)

(xtts-env) C:\Windows\System32>    if tts is None:
is was unexpected at this time.

(xtts-env) C:\Windows\System32>        logger.error("Failed to load XTTS model. Exiting.")
'logger.error' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        return 1
'return' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    # Process all voice samples with language prefixes
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    en_embeddings = process_directory("english_wavs", tts, "en")
'en_embeddings' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    fr_embeddings = process_directory("french_wavs", tts, "fr")
'fr_embeddings' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    all_embeddings = {**en_embeddings, **fr_embeddings}
'all_embeddings' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    if not all_embeddings:
The syntax of the command is incorrect.
(xtts-env) C:\Windows\System32>        logger.error("No embeddings extracted. Exiting.")
'logger.error' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        return 1
'return' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    # Save and analyze results
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    save_embeddings(all_embeddings)
'save_embeddings' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    # Warn about memory for large datasets
'#' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    n = len(all_embeddings)
'n' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    if n > 100:
> was unexpected at this time.

(xtts-env) C:\Windows\System32>        logger.warning(f"Processing {n} files - similarity matrix will require {n*n*8/1e6:.1f}MB RAM")
'logger.warning' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    similarity_df = calculate_similarity(all_embeddings)
'similarity_df' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    logger.info("\n===== SIMILARITY MATRIX =====")
'logger.info' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    logger.info("\n" + similarity_df.to_string())
'logger.info' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>    logger.info("===== PROCESS COMPLETED SUCCESSFULLY =====")
'logger.info' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    return 0
'return' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>
(xtts-env) C:\Windows\System32>if __name__ == "__main__":
The syntax of the command is incorrect.
(xtts-env) C:\Windows\System32>    try:
'try:' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        sys.exit(main())
'sys.exit' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>    except Exception as e:
'except' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        logger.exception("Fatal error:")
'logger.exception' is not recognized as an internal or external command,
operable program or batch file.

(xtts-env) C:\Windows\System32>        sys.exit(1)