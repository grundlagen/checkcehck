from __future__ import annotations

"""Utilities for estimating recursive linguistic complexity.

This module exposes a single function :func:`cort_score` which computes a
light‑weight CORT (complexity/recursion/entropy) score for a piece of text.
The implementation intentionally avoids external dependencies so that it can
run in constrained environments.

The heuristic combines two components:

* Token entropy – Shannon entropy over lower‑cased word tokens.
* Syllable variance – variance of a naive syllable count per token.

Both components are normalised to roughly ``[0, 1]`` and averaged.  The score
is not meant to be linguistically rigorous, merely a rough proxy for how
"complex" or "recursive" a candidate sentence is.
"""

from collections import Counter
import math
import re

__all__ = ["cort_score"]


_vowel_re = re.compile(r"[aeiouy]+", re.I)
_word_re = re.compile(r"\b\w+\b", re.U)


def _syllable_count(word: str) -> int:
    """Very small heuristic syllable counter.

    This simply counts groups of consecutive vowels.  Every word has at least
    one syllable.
    """

    groups = _vowel_re.findall(word)
    return max(1, len(groups))


def cort_score(text: str) -> float:
    """Return a CORT complexity score for ``text``.

    Args:
        text: Input string.

    Returns:
        A float in ``[0, 1]`` where higher values roughly indicate greater
        lexical/phonetic complexity.
    """

    text = text.strip().lower()
    if not text:
        return 0.0

    tokens = _word_re.findall(text)
    n = len(tokens)
    if n == 0:
        return 0.0

    counts = Counter(tokens)
    # Token entropy normalised by max possible entropy (log2(n)).
    ent = -sum((c / n) * math.log(c / n, 2) for c in counts.values())
    max_ent = math.log(n, 2) if n > 1 else 1.0
    norm_ent = ent / max_ent

    # Syllable variance normalised by square of max syllable count.
    syll_counts = [_syllable_count(tok) for tok in tokens]
    mean = sum(syll_counts) / n
    var = sum((s - mean) ** 2 for s in syll_counts) / n
    max_syll = max(syll_counts) or 1
    norm_var = var / (max_syll ** 2)

    return max(0.0, min(1.0, (norm_ent + norm_var) / 2))
