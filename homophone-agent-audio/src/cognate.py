"""
cognate.py
~~~~~~~~~~~

This module provides a lightweight heuristic to detect whether a
proposed French word is a likely cognate or orthographic borrowing of
an English source word.  Cognate detection is important for
homophonic translation because some trivial matches (e.g. "diane"
for "jean") are orthographically similar in both languages and
should be penalized or excluded.  The heuristic is based on
character‑level Jaccard similarity after stripping accents and
non‑alphabetic characters.
"""

from __future__ import annotations
from typing import Set
import unicodedata
import re

def _norm(s: str) -> str:
    """Lowercase and strip diacritics and non‑letters."""
    # Normalize to NFKD and remove diacritics
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # Keep only alphabetic characters
    s = re.sub(r"[^a-zA-Z]", "", s.lower())
    return s

def jaccard(a: str, b: str) -> float:
    """Compute the Jaccard similarity between sets of characters."""
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def is_cognateish(en: str, fr: str, thresh: float = 0.85) -> bool:
    """Return True if ``fr`` looks like a cognate of ``en``."""
    a = _norm(en)
    b = _norm(fr)
    if not a or not b:
        return False
    return jaccard(a, b) >= thresh