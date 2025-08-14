"""ml_judge
=================

Experimental machine-learning based judge that blends multiple
signal modalities to score how well a candidate phrase matches the
source phrase. The judge optionally performs three checks:

* **TTS reconfirmation** – synthesize the candidate text, run speech to
  text, and verify the transcript resembles the source text.
* **Waveform similarity** – synthesize both the source and the
  candidate then measure the similarity between the resulting waveforms.
* **Embedding comparison** – compute semantic similarity between the
  source and candidate texts using sentence embeddings when available.

Each sub‑score is normalized to the range ``[0, 1]`` and the final
``score`` is the unweighted average of all enabled components. When the
required backend or library for a component is unavailable, that
component contributes ``0.0`` but the function still succeeds. The
module is intentionally lightweight and purely function based so that
callers may swap in alternative TTS/ASR/embedding models simply by
replacing the helper functions imported here.

This implementation relies on the stub helpers in
``src.audio_helpers``. For meaningful scores you must override
``synthesize_audio`` and ``speech_to_text`` with real services (e.g.
OpenAI's APIs, local models, etc.). Likewise the default embedding
model comes from ``sentence_transformers``; you can replace the
``semantic_similarity`` helper with any embedding backend that returns a
``[0, 1]`` similarity.
"""

from __future__ import annotations

from typing import Dict
import io
import logging

from .embedding import semantic_similarity
from .audio_helpers import synthesize_audio, heard_as_bonus

# Optional imports for waveform similarity. If unavailable we fall back to
# returning 0.0 for that component.
try:  # pragma: no cover - optional dependency
    import numpy as _np
    import soundfile as _sf
except Exception:  # pragma: no cover - optional dependency missing
    _np = None  # type: ignore
    _sf = None  # type: ignore
    logging.getLogger(__name__).info(
        "soundfile/numpy not available; waveform similarity disabled"
    )


def _waveform_similarity(a_bytes: bytes, b_bytes: bytes) -> float:
    """Compute a crude cosine similarity between two waveforms."""

    if not a_bytes or not b_bytes or _sf is None or _np is None:
        return 0.0
    try:  # pragma: no cover - heavy I/O
        a_wave, a_sr = _sf.read(io.BytesIO(a_bytes))
        b_wave, b_sr = _sf.read(io.BytesIO(b_bytes))
        # Resample to the lower sample rate if necessary
        if a_sr != b_sr:
            target = min(a_sr, b_sr)
            a_wave = _resample(a_wave, a_sr, target)
            b_wave = _resample(b_wave, b_sr, target)
        # Flatten multi-channel audio
        if a_wave.ndim > 1:
            a_wave = a_wave.mean(axis=1)
        if b_wave.ndim > 1:
            b_wave = b_wave.mean(axis=1)
        n = min(len(a_wave), len(b_wave))
        if n == 0:
            return 0.0
        a_wave = a_wave[:n]
        b_wave = b_wave[:n]
        # Normalize
        a_wave = a_wave - _np.mean(a_wave)
        b_wave = b_wave - _np.mean(b_wave)
        denom = float(_np.linalg.norm(a_wave) * _np.linalg.norm(b_wave))
        if denom == 0.0:
            return 0.0
        cos = float(_np.dot(a_wave, b_wave) / denom)
        return (cos + 1.0) / 2.0
    except Exception:
        return 0.0


def _resample(samples, orig_sr: int, target_sr: int):
    """Linear resampling helper used when sample rates differ."""
    if _np is None or orig_sr == target_sr:
        return samples
    duration = len(samples) / float(orig_sr)
    x_old = _np.linspace(0, duration, num=len(samples), endpoint=False)
    x_new = _np.linspace(0, duration, num=int(duration * target_sr), endpoint=False)
    return _np.interp(x_new, x_old, samples)


def ml_judge(
    src_text: str,
    candidate_text: str,
    *,
    use_tts: bool = True,
    use_waveform: bool = True,
    use_embedding: bool = True,
) -> Dict[str, float]:
    """Return a blended machine-learning judge score.

    Parameters
    ----------
    src_text: The original source phrase.
    candidate_text: Candidate phrase to evaluate.
    use_tts / use_waveform / use_embedding: Flags controlling which
        components contribute to the final score.

    Returns
    -------
    Dict[str, float]
        Dictionary containing individual component scores (``embedding``,
        ``waveform`` and ``tts_reconfirm``) and the final ``score`` which is
        the arithmetic mean of the enabled components. Values are always
        between 0 and 1.
    """

    components: Dict[str, float] = {}
    parts: list[float] = []

    if use_embedding:
        emb = semantic_similarity(src_text, candidate_text)
        components["embedding"] = emb
        parts.append(emb)

    # Waveform similarity requires synthesizing both texts
    if use_waveform:
        try:
            src_audio = synthesize_audio(src_text, format="wav")
            cand_audio = synthesize_audio(candidate_text, format="wav")
            wave = _waveform_similarity(src_audio, cand_audio)
        except Exception:
            wave = 0.0
        components["waveform"] = wave
        parts.append(wave)

    if use_tts:
        # ``heard_as_bonus`` performs synthesis and recognition; we interpret
        # any positive bonus as a successful reconfirmation.
        try:
            bonus = heard_as_bonus(src_text, "", candidate_text, "", None)
            tts_score = 1.0 if bonus > 0.0 else 0.0
        except Exception:
            tts_score = 0.0
        components["tts_reconfirm"] = tts_score
        parts.append(tts_score)

    components["score"] = sum(parts) / len(parts) if parts else 0.0
    return components
