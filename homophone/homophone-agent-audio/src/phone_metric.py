"""
phone_metric.py
----------------

This module defines a minimal IPA phone segmentation routine and a
feature‑weighted similarity metric.  It is used both for building
BK‑trees with more linguistically informed distance buckets and as a
fallback phonetic similarity metric when the ``panphon`` library is
unavailable.  The metric assigns lower substitution cost to phones
with similar place or manner of articulation and treats vowels as a
separate class.  Insertions and deletions are given unit cost.

Functions
~~~~~~~~~

``seg_ipa``
    Segment a string of IPA characters into a list of phones.  The
    function uses a hand‑crafted list of multi‑character phones (e.g.
    affricates) to greedily match longest substrings first.

``feature_distance``
    Compute a normalized edit distance between two IPA strings using
    the feature‑weighted substitution cost.  The result is in the
    range [0, 1], where 0 indicates identical and 1 indicates maximally
    different.

``similarity``
    Convenience wrapper returning 1 minus the feature distance.

Notes
~~~~~

The feature sets defined here are coarse and not intended to
perfectly model phonetic realities.  They simply provide a useful
approximation when more sophisticated tools (like PanPhon) are not
available.  You can safely expand the `_PHONE_ORDER`, `_PLACE`, and
`_MANNER` sets if you add new IPA symbols to your lexicon.
"""

from __future__ import annotations

from typing import List
import re

# Ordered list of multi‑character phones to match greedily.  Affricates
# and other multi‑segment phones should appear before their substrings.
_PHONE_ORDER: List[str] = [
    "t͡s", "d͡z", "t͡ʃ", "d͡ʒ", "p͡f",
    "dʒ", "tʃ", "ts", "dz", "pf",
    "ʧ", "ʤ",  # alternative affricate spellings
    "ʃ", "ʒ",
    "ɲ", "ŋ", "ɡ", "ɫ", "ɾ", "ɹ", "ʁ", "x", "ɣ",
    "ɑ", "ɒ", "a", "æ", "ɐ", "ə", "ɛ", "e", "i", "ɪ", "ɨ", "ʏ", "y",
    "o", "ɔ", "u", "ʊ", "œ", "ø", "ɜ", "ɞ", "ʌ", "ɚ", "ɝ",
    "ʔ", "h",
    "b", "p", "d", "t", "g", "k", "q",
    "v", "f", "z", "s", "ʑ", "ɕ", "ç", "ʝ", "m", "n", "l", "r", "j", "w",
]

# Compile a regex that matches any of the above phones at the beginning of a string.
_PHONE_RE = re.compile("|".join(map(re.escape, _PHONE_ORDER)))

def seg_ipa(s: str) -> List[str]:
    """Segment an IPA string into a list of phone symbols.

    The function removes spaces and greedily matches the longest
    multi‑character phones from ``_PHONE_ORDER``.  Characters not in the
    known set are treated as singletons.

    Args:
        s: The IPA string to segment.

    Returns:
        A list of phone strings.
    """
    s = s.replace(" ", "")
    phones: List[str] = []
    i = 0
    while i < len(s):
        m = _PHONE_RE.match(s, i)
        if m:
            phones.append(m.group(0))
            i = m.end()
        else:
            phones.append(s[i])
            i += 1
    return phones

# Coarse feature classes for consonants.  Keys map to sets of IPA symbols.
_VOWELS = set("ɑɒaæɐəɛeiɪɨʏyoɔuʊœøɜɞʌɚɝ")
_PLACE = {
    "labial": set(list("bmpfvwɸβ")),
    "dental": set(list("tdsznlθð")),
    "alveo":  set(list("tdsznlrɾɹ")),
    "postal": set(list("ʃʒʧʤʂʐɕʑ")),
    "palat":  set(list("jçʝɲ")),
    "velar":  set(list("kgxɣŋɫ")),
    "uvular": set(list("qʁ")),
    "glott":  set(list("hʔ")),
}
_MANNER = {
    "stop": set(list("ptkbdgqʔ")),
    "aff":  set(["t͡s", "d͡z", "t͡ʃ", "d͡ʒ", "ts", "dz", "tʃ", "dʒ", "ʧ", "ʤ", "p͡f"]),
    "fric": set(list("fvszʃʒθðxɣçʝɸβh")),
    "nas":  set(list("mnŋɲ")),
    "lat":  set(list("lɫ")),
    "apr":  set(list("rwɹɾjʁ")),
}

def _bucket(phone: str, groups: dict) -> str | None:
    """Return the key of the group containing ``phone``, or ``None``."""
    for k, s in groups.items():
        if phone in s:
            return k
    return None

def phone_sim(p: str, q: str) -> float:
    """Return a similarity score between two phone symbols in [0, 1]."""
    if p == q:
        return 1.0
    pv, qv = p in _VOWELS, q in _VOWELS
    # Vowel–vowel: moderate similarity
    if pv and qv:
        return 0.6
    # Vowel vs consonant: no similarity
    if pv or qv:
        return 0.0
    pp, qp = _bucket(p, _PLACE), _bucket(q, _PLACE)
    pm, qm = _bucket(p, _MANNER), _bucket(q, _MANNER)
    sim = 0.0
    if pp and qp and pp == qp:
        sim += 0.45
    if pm and qm and pm == qm:
        sim += 0.45
    # Voicing similarity (approximate)
    voiced = set("bdgvzʒʝɣʐβ")
    voiceless = set("ptkfsʃçxʂθɸ")
    if (p in voiced and q in voiced) or (p in voiceless and q in voiceless):
        sim += 0.1
    return min(sim, 0.95)

def feature_distance(ipa1: str, ipa2: str) -> float:
    """Compute a normalized feature‑based edit distance between two IPA strings.

    The distance is the Levenshtein distance between the phone lists
    produced by ``seg_ipa``, with substitution cost equal to ``1 − phone_sim``.
    Insertions and deletions cost 1.  The result is divided by the
    maximum length of the phone lists to yield a value in [0, 1].

    Args:
        ipa1: First IPA string.
        ipa2: Second IPA string.
    Returns:
        A float between 0 and 1 representing dissimilarity.
    """
    a = seg_ipa(ipa1)
    b = seg_ipa(ipa2)
    n, m = len(a), len(b)
    # Initialize DP table
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = float(i)
    for j in range(1, m + 1):
        dp[0][j] = float(j)
    # Fill DP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + (1.0 - phone_sim(a[i - 1], b[j - 1]))
            ins = dp[i][j - 1] + 1.0
            dele = dp[i - 1][j] + 1.0
            dp[i][j] = min(sub, ins, dele)
    denom = max(n, m) or 1
    return dp[n][m] / denom

def similarity(ipa1: str, ipa2: str) -> float:
    """Return a similarity score in [0, 1] for two IPA strings."""
    return 1.0 - feature_distance(ipa1, ipa2)