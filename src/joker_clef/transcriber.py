"""Utilities for turning speech clips into text."""

from __future__ import annotations
from pathlib import Path


def transcribe_clip(clip_path: str | Path) -> str:
    """Transcribe an audio clip using a speech-recognition engine.

    Parameters
    ----------
    clip_path:
        Path to the audio clip to transcribe.

    Returns
    -------
    str
        The recognised text.

    Notes
    -----
    This function requires the :mod:`speech_recognition` package and an
    available backend such as PocketSphinx or DeepSpeech. The heavy lifting
    is intentionally imported lazily so that tests can mock the function
    without needing those optional dependencies installed.
    """
    # Import inside the function to keep optional dependencies lazy.
    import speech_recognition as sr  # type: ignore

    recognizer = sr.Recognizer()
    audio_file = Path(clip_path)
    with sr.AudioFile(str(audio_file)) as source:
        audio_data = recognizer.record(source)
    # PocketSphinx runs offline and avoids network access during demos.
    return recognizer.recognize_sphinx(audio_data)
