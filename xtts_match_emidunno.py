import os
import sys
import subprocess
import logging
import glob
import importlib.util
import librosa
from pydub import AudioSegment
# Configure FFmpeg paths - MORE ROBUST VERSION
ffmpeg_path = r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"
ffprobe_path = r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffprobe.exe"

def configure_ffmpeg():
    """Ensure FFmpeg is properly configured and accessible"""
    try:
        # Set paths
        if os.path.exists(ffmpeg_path):
            AudioSegment.converter = ffmpeg_path
            AudioSegment.ffprobe = ffprobe_path
            os.environ['PATH'] += os.pathsep + os.path.dirname(ffmpeg_path)
        else:
            raise RuntimeError(f"FFmpeg not found at {ffmpeg_path}")
        
        # Verify
        from pydub.utils import which
        if not which(AudioSegment.converter):
            raise RuntimeError("FFmpeg not found in system PATH")
        
        # Test conversion
        test_audio = AudioSegment.silent(duration=1000)
        test_audio.export(os.devnull, format="wav")
        return True
        
    except Exception as e:
        logger.error(f"FFmpeg configuration failed: {str(e)}")
        return False

if not configure_ffmpeg():
    sys.exit(1)
    
DEPENDENCY_MAP = {
    "TTS": "TTS",
    "librosa": "librosa",
    "scipy": "scipy",
    "numpy": "numpy",
    "pandas": "pandas",
    "torch": "torch",
    "pydub": "pydub",  # NEW DEPENDENCY FOR MP3 SUPPORT
    "ffmpeg-python": "ffmpeg"  # Required for pydub to handle MP3
}


# Dependency check must come FIRST
def check_dependencies():
    """Install missing dependencies with explicit mapping"""
    print("===== DEPENDENCY CHECK =====")
    installed_anything = False
    
    for pkg, import_name in DEPENDENCY_MAP.items():
        # Check using importlib to handle dotted paths
        spec = importlib.util.find_spec(import_name.split('.')[0])
        if spec is None:
            print(f"Installing {pkg}...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"Installed {pkg}")
                installed_anything = True
            except Exception as e:
                print(f"Failed to install {pkg}: {str(e)}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"{pkg} already installed")
    
    if installed_anything:
        print("\n!!! RESTART REQUIRED !!! New dependencies installed.", file=sys.stderr)
        sys.exit(86)  # Special exit code for restart

# Check dependencies before any other imports
check_dependencies()

# Now safe to import everything
import numpy as np
import pandas as pd
import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs
from pydub import AudioSegment  # NEW IMPORT FOR MP3 SUPPORT
import librosa  # Will use for audio processing
import soundfile as sf  # Keep for WAV files

# Initialize torch serialization safely
torch.serialization.add_safe_globals([
    XttsConfig, 
    XttsAudioConfig,
    BaseDatasetConfig,
    XttsArgs  # ADDED TO FIX "Unsupported global" ERROR
])

# Enhanced logging configuration
class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            if line.strip():
                self.logger.log(self.log_level, line.rstrip())
    
    def flush(self):
        # Robust flush handling
        for handler in self.logger.handlers:
            flush_func = getattr(handler, 'flush', None)
            if callable(flush_func):
                flush_func()

# Configure dual logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("xtts_progress.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('XTTS')
sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

def print_system_info():
    """Display system and package info"""
    info = [
        "\n=== SYSTEM INFORMATION ===",
        f"Python: {sys.version.split()[0]}",
        f"Platform: {sys.platform}",
        f"Working Dir: {os.getcwd()}",
        "\n=== PACKAGE VERSIONS ==="
    ]
    
    for pkg in DEPENDENCY_MAP.values():
        try:
            mod = __import__(pkg)
            info.append(f"{pkg}: {getattr(mod, '__version__', 'unknown')}")
        except ImportError:
            info.append(f"{pkg}: not available")
    
    logger.info("\n".join(info))

def load_xtts_model(force_cpu=False):
    """Load model with secure weights handling"""
    try:
        logger.info("Initializing XTTS model...")
        device = "cpu" if force_cpu else "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
            gpu=(device == "cuda")
        )
        logger.info(f"XTTS loaded on {device.upper()}")
        return tts
    except Exception as e:
        logger.error(f"Model load failed: {str(e)}")
        return None

# FIXED INDENTATION - moved this function OUTSIDE load_xtts_model
def convert_mp3_to_wav(mp3_path, temp_files):
    """Convert MP3 to temporary WAV file with robust path handling"""
    try:
        mp3_path = os.path.normpath(mp3_path)
        if not os.path.exists(mp3_path):
            raise FileNotFoundError(f"MP3 file not found: {mp3_path}")
            
        # Create temporary WAV path in system temp directory
        temp_dir = os.path.join(os.environ.get('TEMP', os.getcwd()), 'xtts_temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        base_name = os.path.basename(mp3_path)
        wav_path = os.path.join(temp_dir, f"{os.path.splitext(base_name)[0]}_temp.wav")
        
        # Convert with FFmpeg
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav", parameters=["-ac", "1", "-ar", "22050"])
        
        # Track the temp file
        temp_files.append(wav_path)
        return wav_path
        
    except Exception as e:
        logger.error(f"MP3 conversion failed for {mp3_path}: {str(e)}")
        return None

@torch.inference_mode()
def extract_speaker_embedding(audio_path, tts, temp_files):
    """Extract speaker embedding with MP3 support"""
    try:
        logger.info(f"Processing: {os.path.basename(audio_path)}")
        
        temp_file = None
        try:
            if audio_path.lower().endswith('.mp3'):
                temp_file = convert_mp3_to_wav(audio_path, temp_files)
                if not temp_file:
                    return None
                audio_path = temp_file
            
            audio, sr = librosa.load(audio_path, sr=22050, mono=True)
            logger.info(f"Audio loaded: {len(audio)/sr:.2f}s @ {sr}Hz")
            
            try:
                speaker_latent, _, _ = tts.tts_model.get_conditioning_latents(audio, sr)
            except AttributeError:
                speaker_latent, _, _ = tts.synthesizer.tts_model.get_conditioning_latents(audio, sr)
        for audio_path in audio_files:
            try:
                embedding = extract_speaker_embedding(audio_path, tts, temp_files)
                if embedding is not None:
                    rel_path = f"{lang_prefix}/{os.path.relpath(audio_path, directory)}"
                    embeddings[rel_path] = embedding
            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Directory processing failed: {str(e)}")
        
    return embeddings


def main():
    """Main workflow with error handling"""
    temp_files = []  # Now properly tracks all temp files
    
    def cleanup():
        """Clean up temporary files"""
        cleaned = 0
        for file in temp_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
                    cleaned += 1
            except Exception as e:
                logger.error(f"Cleanup failed for {file}: {str(e)}")
        logger.info(f"Cleaned up {cleaned}/{len(temp_files)} temporary files")
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        import atexit
        atexit.register(cleanup)
        
        logger.info("===== PHONETIC EMBEDDING EXTRACTION STARTED =====")
        print_system_info()
        
        tts = load_xtts_model(force_cpu=False)
        if tts is None:
            return 1
        
        en_embeddings = process_directory(
            r"C:\Users\Lenovo\Desktop\wav_english", tts, "en", temp_files)
        fr_embeddings = process_directory(
            r"C:\Users\Lenovo\Desktop\wav_french", tts, "fr", temp_files)
        
        if not (en_embeddings or fr_embeddings):
            logger.error("No embeddings extracted. Exiting.")
            return 1
        
        save_embeddings({**en_embeddings, **fr_embeddings})
    return embeddings

def save_embeddings(embeddings, filename="voice_embeddings.npz"):
    """Save embeddings as float32 arrays"""
    np.savez(
        filename,
        **{k: v.astype(np.float32) for k, v in embeddings.items()}
    )
    logger.info(f"Saved embeddings to {filename}")

def calculate_similarity(embeddings):
    """Vectorized cosine similarity calculation"""
    filenames = list(embeddings.keys())
    embeddings_arr = np.array([emb.flatten() for emb in embeddings.values()])
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Prevent division by zero
    embeddings_norm = embeddings_arr / norms
    
    # Vectorized similarity matrix
    similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
    return pd.DataFrame(similarity_matrix, index=filenames, columns=filenames)

def main():
    """Main workflow with error handling"""
    temp_files = []  # Now properly tracks all temp files
    
    def cleanup():
        """Clean up temporary files"""
        cleaned = 0
        for file in temp_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
                    cleaned += 1
            except Exception as e:
                logger.error(f"Cleanup failed for {file}: {str(e)}")
        logger.info(f"Cleaned up {cleaned}/{len(temp_files)} temporary files")
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        import atexit
        atexit.register(cleanup)
        
        logger.info("===== PHONETIC EMBEDDING EXTRACTION STARTED =====")
        print_system_info()
        
        tts = load_xtts_model(force_cpu=False)
        if tts is None:
            return 1
        
        en_embeddings = process_directory(
            r"C:\Users\Lenovo\Desktop\wav_english", tts, "en", temp_files)
        fr_embeddings = process_directory(
            r"C:\Users\Lenovo\Desktop\wav_french", tts, "fr", temp_files)
        
        if not (en_embeddings or fr_embeddings):
            logger.error("No embeddings extracted. Exiting.")
            return 1
        
        all_embeddings = {**en_embeddings, **fr_embeddings}
        save_embeddings(all_embeddings)
        
        # Warn about memory for large datasets
        n = len(all_embeddings)
        if n > 100:
            logger.warning(f"Processing {n} files - similarity matrix will require {n*n*8/1e6:.1f}MB RAM")
        
        similarity_df = calculate_similarity(all_embeddings)
        
        logger.info("\n===== SIMILARITY MATRIX =====")
        logger.info("\n" + similarity_df.to_string())
        
        logger.info("===== PROCESS COMPLETED SUCCESSFULLY =====")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        cleanup()
        return 0
    except Exception as e:
        logger.exception("Fatal error:")
        cleanup()
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.exception("Fatal error:")
        sys.exit(1)