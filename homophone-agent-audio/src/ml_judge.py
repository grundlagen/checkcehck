"""ML-based judge utilities.

This module implements :func:`ml_judge` which combines several machine
learning signals to assess how well a candidate translation matches the
source phrase.  The three signals are:

* **TTS reconfirmation** – synthesize the candidate text, run speech-to-text
  on the audio and check whether the recognizer hears something similar to
  the source text.
* **Waveform similarity** – synthesize both the source and candidate and
  compare their audio waveforms directly.
* **Embedding similarity** – compute a semantic similarity score between the
  source and candidate texts using sentence embeddings or a fallback
  heuristic.

Each component is optional and can be toggled via flags.  The final score is
the unweighted average of the enabled components.  The implementation is
backend-agnostic: real TTS/ASR and embedding models can be injected by
replacing the callable hooks exposed at module level.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional
import io
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hooks for external models
#
# The default implementations come from ``src.audio_helpers`` and
# ``src.embedding``.  Clients can monkey patch these callables to integrate
# custom backends (e.g. OpenAI APIs or local models).

from .audio_helpers import synthesize_audio as _synthesize_audio
from .audio_helpers import heard_as_bonus as _heard_as_bonus
from .embedding import semantic_similarity as _semantic_similarity
from .phone_distance import PhoneDistance

# Exposed hooks – replace these with your own functions if desired.
TTS_FN: Callable[[str, str], bytes] = _synthesize_audio
EMBEDDING_FN: Callable[[str, str], float] = _semantic_similarity


def set_tts_backend(fn: Callable[[str, str], bytes]) -> None:
    """Replace the text-to-speech synthesis function used by :func:`ml_judge`.

    Parameters
    ----------
    fn:
        Callable that accepts ``(text, format)`` and returns audio bytes.
    """

    global TTS_FN
    TTS_FN = fn


def set_embedding_backend(fn: Callable[[str, str], float]) -> None:
    """Replace the embedding similarity function used by :func:`ml_judge`."""

    global EMBEDDING_FN
    EMBEDDING_FN = fn


# ---------------------------------------------------------------------------
# Internal helpers


def _waveform_similarity(aud_a: bytes, aud_b: bytes) -> float:
    """Compute a simple waveform similarity between two audio clips.

    The current implementation decodes the audio using :mod:`soundfile` and
    computes a normalized cross-correlation.  If decoding fails or the
    dependency is missing, ``0.0`` is returned.
    """

    if not aud_a or not aud_b:
        return 0.0
    try:
        import numpy as np
        import soundfile as sf

        a_wave, sr_a = sf.read(io.BytesIO(aud_a))
        b_wave, sr_b = sf.read(io.BytesIO(aud_b))
        if sr_a != sr_b or len(a_wave) == 0 or len(b_wave) == 0:
            return 0.0
        # Truncate to the shorter length
        n = min(len(a_wave), len(b_wave))
        a_wave = a_wave[:n]
        b_wave = b_wave[:n]
        # Zero-mean
        a_wave = a_wave - a_wave.mean()
        b_wave = b_wave - b_wave.mean()
        # Normalized correlation
        denom = float(np.linalg.norm(a_wave) * np.linalg.norm(b_wave))
        if denom == 0.0:
            return 0.0
        corr = float(np.dot(a_wave, b_wave) / denom)
        return (corr + 1.0) / 2.0  # map [-1, 1] -> [0, 1]
    except Exception as exc:
        logger.debug("waveform similarity failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Public API


def ml_judge(
    src_text: str,
    candidate_text: str,
    *,
    use_tts: bool = True,
    use_waveform: bool = True,
    use_embedding: bool = True,
) -> Dict[str, float]:
    """Fuse multiple ML signals into a single score.

    Parameters
    ----------
    src_text:
        Original source phrase.
    candidate_text:
        Candidate translation to evaluate.
    use_tts, use_waveform, use_embedding:
        Flags controlling which components to compute.

    Returns
    -------
    Dict[str, float]
        Dictionary containing individual component scores (``tts``,
        ``waveform`` and ``embedding``) along with the final ``score`` key.
    """

    scores: Dict[str, float] = {}
    components = []

    if use_tts:
        try:
            # ``heard_as_bonus`` returns a positive value when ASR reconfirms
            # the phrase; treat any positive bonus as a successful match.
            bonus = _heard_as_bonus(src_text, "", candidate_text, "", PhoneDistance(), bonus=1.0)
            tts_score = 1.0 if bonus > 0.0 else 0.0
        except Exception as exc:
            logger.debug("TTS reconfirmation failed: %s", exc)
            tts_score = 0.0
        scores["tts"] = tts_score
        components.append(tts_score)

    if use_waveform:
        try:
            aud_src = TTS_FN(src_text, format="wav")
            aud_cand = TTS_FN(candidate_text, format="wav")
        except Exception as exc:
            logger.debug("TTS synthesis for waveform comparison failed: %s", exc)
            aud_src = b""
            aud_cand = b""
        wave_score = _waveform_similarity(aud_src, aud_cand)
        scores["waveform"] = wave_score
        components.append(wave_score)

    if use_embedding:
        try:
            emb_score = EMBEDDING_FN(src_text, candidate_text)
        except Exception as exc:
            logger.debug("Embedding similarity failed: %s", exc)
            emb_score = 0.0
        scores["embedding"] = emb_score
        components.append(emb_score)

    scores["score"] = sum(components) / len(components) if components else 0.0
    return scores

