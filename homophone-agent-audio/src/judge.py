"""
judge.py
---------

This module provides a small helper to compute phonetic, semantic and
fluency scores for a candidate translation and to package them with a
short rationale. It centralizes the core scoring logic so that the
scoring and co‑optimization loop can be kept uncluttered.

The judge function returns normalized values in the range [0, 1] and a
concise string summarizing the component scores. It makes no attempt
to explain how the scores were derived beyond reporting them; it
explicitly avoids exposing chain‑of‑thought reasoning in keeping with
OpenAI’s best practices.

Usage:

    from .phone_distance import PhoneDistance
    from .judge import judge

    pdist = PhoneDistance()
    result = judge(src_text, src_ipa, A_text, B_text, B_ipa, pdist)
    print(result)
"""

from __future__ import annotations

from typing import Dict

from .embedding import semantic_similarity
from .orchestrator import fluency_score
from .cognate import is_cognateish
from .phone_distance import PhoneDistance
from .cort import cort_score


def judge(
    src_text: str,
    src_ipa: str,
    A_text: str,
    B_text: str,
    B_ipa: str,
    phone_dist: PhoneDistance,
    compute_complexity: bool = False,
) -> Dict[str, float | str]:
    """
    Compute component scores and a rationale for a candidate translation.

    Args:
        src_text: The original source string.
        src_ipa: The IPA representation of the source (can be empty).
        A_text: The literal translation text (used for semantic anchor).
        B_text: The candidate homophonic translation text.
        B_ipa: The IPA for the candidate translation.
        phone_dist: A PhoneDistance instance for phonetic similarity.
        compute_complexity: Whether to compute the optional CORT
            complexity metric for ``B_text``.

    Returns:
        A dictionary with keys ``phonetic``, ``semantic``, ``fluency`` and
        ``rationale``. Scores are floats in [0, 1]. The rationale is a
        concise human‑readable string summarizing the scores.
    """
    # Semantic similarity between the literal translation and the candidate
    sem = semantic_similarity(A_text, B_text)
    # Fluency heuristic
    flu = fluency_score(B_text)
    # Phonetic similarity: compare candidate IPA to source IPA
    phon = phone_dist.similarity(src_ipa, B_ipa) if src_ipa and B_ipa else 0.0
    # Apply a cognate penalty: if any token in B looks like the corresponding token in src
    # (character overlap ≥ 0.85), reduce fluency slightly.  This discourages trivial
    # orthographic matches (e.g. "Diane" for "jean").
    tokens_src = src_text.split()
    tokens_B = B_text.split()
    cog_pen = 0.0
    # Only compare up to the shorter length to avoid index errors
    for a, b in zip(tokens_src, tokens_B):
        if is_cognateish(a, b, thresh=0.85):
            cog_pen += 0.05
    # Cap penalty at 0.3
    cog_pen = min(cog_pen, 0.3)
    flu_adj = max(0.0, flu - cog_pen)
    # Optional complexity score
    complexity = cort_score(B_text) if compute_complexity else 0.0
    # Construct rationale string
    rationale = (
        f"phonetic {phon:.3f}; semantic {sem:.3f}; fluency {flu_adj:.3f}"
    )
    if compute_complexity:
        rationale += f"; complexity {complexity:.3f}"
    result: Dict[str, float | str] = {
        "phonetic": phon,
        "semantic": sem,
        "fluency": flu_adj,
        "rationale": rationale,
    }
    if compute_complexity:
        result["complexity"] = complexity
    return result

# -----------------------------------------------------------------------------
# Additional judges
#
# The CLEF JOKER pipeline uses multiple evaluators beyond basic phonetic,
# semantic and fluency checks.  One useful evaluator is a TTS reconfirmation
# judge: it synthesizes the candidate text into audio, runs speech
# recognition, and checks whether the recognized transcript matches the
# source sufficiently well.  This implementation relies on the helper
# ``heard_as_bonus`` from ``src.audio_helpers``.  Because the default
# environment lacks real TTS/ASR backends, the function currently returns
# zero unless clients override the audio helpers.

from .audio_helpers import heard_as_bonus  # type: ignore

def judge_tts(
    src_text: str,
    src_ipa: str,
    candidate_text: str,
    candidate_ipa: str,
    phone_dist: PhoneDistance,
    bonus_threshold: float = 0.05,
) -> float:
    """Evaluate whether a candidate sounds sufficiently like the source via TTS/ASR.

    This helper attempts to synthesize the candidate text into audio and
    transcribe it back via ASR.  If the transcript matches the source
    reasonably well (as determined by ``heard_as_bonus``), it returns 1.0.
    Otherwise it returns 0.0.  The default thresholds used by
    ``heard_as_bonus`` are conservative; callers can adjust
    ``bonus_threshold`` to tweak the sensitivity.

    Args:
        src_text: The original source string.
        src_ipa: The IPA representation of the source (can be empty).
        candidate_text: The candidate translation text (French).
        candidate_ipa: The IPA for the candidate translation.
        phone_dist: A PhoneDistance instance used internally by
            ``heard_as_bonus`` when computing phonetic similarity.
        bonus_threshold: The bonus value requested; positive values
            indicate that ASR must return this bonus or higher to pass.

    Returns:
        1.0 if the ASR reconfirmation passes; 0.0 otherwise.
    """
    try:
        bonus = heard_as_bonus(
            src_text,
            src_ipa,
            candidate_text,
            candidate_ipa,
            phone_dist,
            bonus=bonus_threshold,
        )
        return 1.0 if bonus and bonus > 0.0 else 0.0
    except Exception:
        # On any error, treat as failure (no ASR reconfirmation)
        return 0.0
