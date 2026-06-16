"""
phon.py — self-contained IPA normalisation, segmentation and feature-weighted
phonetic similarity.

This is a dependency-free distillation of the metric that already lives in
``homophone-agent-audio/src/phone_metric.py``.  It is vendored here so the
``meaning_map`` routine can run anywhere (no panphon / rapidfuzz / numpy)
while staying faithful to the established scoring philosophy: substitutions
between articulatorily similar phones are cheap, between distant phones are
expensive, and vowels live in their own class.

Two public entry points matter for the Map of Meaning:

``normalise_ipa``   -> a stress/length/diacritic-stripped form used as a
                       *bucket key* so candidate pairs can be found without an
                       O(N*M) sweep over the two lexica.
``similarity``      -> a [0,1] phonetic similarity over two raw IPA strings,
                       used to *score* candidates inside a bucket.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

# --- characters that carry suprasegmental / connective information only -------
# Stress marks, length, tie bars, zero-width joiners, syllable dots, spaces.
_STRIP = {
    "ˈ",  # ˈ primary stress
    "ˌ",  # ˌ secondary stress
    "ː",  # ː length
    "ˑ",  # ˑ half length
    "͡",  # ͡ tie bar (combining)
    "͜",  # ͜ tie bar below
    "‍",  # zero-width joiner (appears in the en lexique)
    "‌",  # zero-width non-joiner
    ".",       # syllable break
    " ",
    "'",
    "ˀ",
}

# Greedy multi-char phones (affricates etc.) — longest first.
_PHONE_ORDER: List[str] = [
    "t͡s", "d͡z", "t͡ʃ", "d͡ʒ", "p͡f",
    "dʒ", "tʃ", "ts", "dz", "pf", "ʧ", "ʤ",
    "ʃ", "ʒ", "ɲ", "ŋ", "ɡ", "ɫ", "ɾ", "ɹ", "ʁ", "x", "ɣ",
    "ɑ", "ɒ", "a", "æ", "ɐ", "ə", "ɛ", "e", "i", "ɪ", "ɨ", "ʏ", "y",
    "o", "ɔ", "u", "ʊ", "œ", "ø", "ɜ", "ɞ", "ʌ", "ɚ", "ɝ",
    "ʔ", "h",
    "b", "p", "d", "t", "g", "k", "q",
    "v", "f", "z", "s", "ʑ", "ɕ", "ç", "ʝ", "m", "n", "l", "r", "j", "w",
]
_PHONE_RE = re.compile("|".join(map(re.escape, _PHONE_ORDER)))

_VOWELS = set("ɑɒaæɐəɛeiɪɨʏyoɔuʊœøɜɞʌɚɝ")
_PLACE = {
    "labial": set("bmpfvwɸβ"),
    "dental": set("tdsznlθð"),
    "alveo": set("tdsznlrɾɹ"),
    "postal": set("ʃʒʧʤʂʐɕʑ"),
    "palat": set("jçʝɲ"),
    "velar": set("kgɡxɣŋɫ"),
    "uvular": set("qʁ"),
    "glott": set("hʔ"),
}
_MANNER = {
    "stop": set("ptkbdgɡqʔ"),
    "aff": {"t͡s", "d͡z", "t͡ʃ", "d͡ʒ", "ts", "dz", "tʃ", "dʒ", "ʧ", "ʤ", "p͡f"},
    "fric": set("fvszʃʒθðxɣçʝɸβh"),
    "nas": set("mnŋɲ"),
    "lat": set("lɫ"),
    "apr": set("rwɹɾjʁ"),
}
_VOICED = set("bdgɡvzʒʝɣʐβmnŋɲlɫrɹjw")
_VOICELESS = set("ptkfsʃçxʂθɸh")


def normalise_ipa(s: str) -> str:
    """Strip suprasegmentals and combining diacritics; return a bare phone string.

    French nasal vowels (ɑ̃, ɔ̃ …) lose their tilde here — that is intentional
    for *bucketing* only.  Fine-grained scoring is done on the raw string.
    """
    out = []
    for ch in s:
        if ch in _STRIP:
            continue
        # drop combining marks (tilde, etc.) for the coarse key
        if unicodedata.combining(ch):
            continue
        out.append(ch)
    return "".join(out)


def seg_ipa(s: str) -> List[str]:
    s = normalise_ipa(s)
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


def _bucket(phone: str, groups: dict) -> str | None:
    for k, members in groups.items():
        if phone in members:
            return k
    return None


def phone_sim(p: str, q: str) -> float:
    if p == q:
        return 1.0
    pv, qv = p in _VOWELS, q in _VOWELS
    if pv and qv:
        return 0.6
    if pv or qv:
        return 0.0
    pp, qp = _bucket(p, _PLACE), _bucket(q, _PLACE)
    pm, qm = _bucket(p, _MANNER), _bucket(q, _MANNER)
    sim = 0.0
    if pp and qp and pp == qp:
        sim += 0.45
    if pm and qm and pm == qm:
        sim += 0.45
    if (p in _VOICED and q in _VOICED) or (p in _VOICELESS and q in _VOICELESS):
        sim += 0.1
    return min(sim, 0.95)


def feature_distance(ipa1: str, ipa2: str) -> float:
    a, b = seg_ipa(ipa1), seg_ipa(ipa2)
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 0.0
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = float(i)
    for j in range(1, m + 1):
        dp[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + (1.0 - phone_sim(a[i - 1], b[j - 1]))
            dp[i][j] = min(sub, dp[i][j - 1] + 1.0, dp[i - 1][j] + 1.0)
    return dp[n][m] / (max(n, m) or 1)


def similarity(ipa1: str, ipa2: str) -> float:
    return 1.0 - feature_distance(ipa1, ipa2)


def signature(ipa: str) -> str:
    """A coarse phonetic bucket key.

    Two words can only be near-homophones if they share: vowel count, and the
    *class* of their first and last phone.  This shrinks the candidate space
    from O(N*M) to the sum over shared buckets, which is tractable on the full
    65k/246k lexica.
    """
    phones = seg_ipa(ipa)
    if not phones:
        return "∅"
    vcount = sum(1 for p in phones if p in _VOWELS)

    def cls(p: str) -> str:
        if p in _VOWELS:
            return "V"
        return _bucket(p, _MANNER) or "?"

    return f"{vcount}|{cls(phones[0])}|{cls(phones[-1])}"
