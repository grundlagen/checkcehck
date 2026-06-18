"""
ipa_norm.py
~~~~~~~~~~~

IPA normalization for fair phonetic comparison.

The raw IPA strings in our lexica (``lexique_en.tsv``, ``lexique.tsv``) and
those produced by espeak/phonemizer carry **suprasegmental decoration** that
hurts naive alignment: primary/secondary stress marks (``ˈ`` ``ˌ``), length
marks (``ː`` ``ˑ``), syllable boundaries, linking ties and tie bars over
affricates.  Two pronunciations that a human hears as identical — e.g. English
``see`` /sˈiː/ and French ``si`` /si/ — score far apart unless these marks are
removed first.

This module strips that decoration so the downstream feature-weighted edit
distance in :mod:`phone_metric` operates on the bare segmental skeleton.  It is
deliberately conservative: it removes only diacritics/marks that do not change
the identity of a consonant or vowel for our coarse feature model.  Nasal
vowels keep their base vowel quality (the tilde is a combining mark and is
dropped), which is the right call for a coarse cross-lingual matcher.

Empirically this lifts honest sound matches substantially, e.g.::

    see /sˈiː/ ~ si /si/        0.50 -> 1.00
    envy /ˈɛnvi/ ~ envie /ɑ̃vi/  0.52 -> 0.65
"""

from __future__ import annotations

import unicodedata

# Suprasegmental / boundary symbols that should never count as phones.
_STRIP_CHARS = set("ˈˌːˑ‿‖|.ˀ⁀-")
# Tie bars (combining and spacing variants) joining e.g. affricates t͡ʃ.
_TIE_CHARS = set("͜͡‿")  # ͡ ͜ ‿


def norm_ipa(s: str) -> str:
    """Return a bare segmental IPA string suitable for phonetic comparison.

    Removes stress, length, syllable/linking marks, tie bars and all
    combining diacritics (nasalisation, etc.), and drops whitespace.  The
    surviving characters are the base vowels and consonants that the coarse
    feature model in :mod:`phone_metric` understands.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    out = []
    for ch in s:
        if ch in _STRIP_CHARS or ch in _TIE_CHARS:
            continue
        if unicodedata.combining(ch):
            continue
        if ch.isspace():
            continue
        out.append(ch)
    return "".join(out)
