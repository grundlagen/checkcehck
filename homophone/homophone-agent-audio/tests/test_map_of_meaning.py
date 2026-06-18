"""Tests for the Map of Meaning engine.

These exercise the engine on small in-memory fixtures (no data files needed)
plus, when the real data is present, a couple of headline end-to-end checks.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.ipa_norm import norm_ipa  # noqa: E402
from src.map_of_meaning import MapOfMeaning  # noqa: E402


# --------------------------------------------------------------------------- #
# ipa_norm
# --------------------------------------------------------------------------- #
def test_norm_strips_stress_and_length():
    assert norm_ipa("sˈiː") == "si"
    assert norm_ipa("/sˈiː/".strip("/")) == "si"


def test_norm_strips_tie_bars_and_combining():
    # nasal vowel: combining tilde dropped, base vowel kept
    assert norm_ipa("ɑ̃vi") == "ɑvi"
    # affricate tie bar removed
    assert norm_ipa("t͡ʃat") == "tʃat"


# --------------------------------------------------------------------------- #
# engine on fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def eng():
    en = {"soup": "sˈuːp", "debt": "dˈɛt", "envy": "ˈɛnvi", "love": "lˈʌv"}
    fr = {"soupe": "sup", "dette": "dɛt", "envie": "ɑ̃vi", "louve": "luv"}
    gloss = {
        "soupe": ["soup"],
        "dette": ["debt"],
        "envie": ["envy", "jealousy"],
        "louve": ["she-wolf"],
        "amour": ["love", "affection"],
    }
    return MapOfMeaning(en_lex=en, fr_lex=fr, glosses=gloss)


def test_sound_sim_identical_after_norm(eng):
    assert eng.sound_sim("sˈuːp", "sup") == pytest.approx(1.0)


def test_golden_matches_finds_cognate_and_gem(eng):
    matches = {(m.fr, m.en): m for m in eng.golden_matches(min_sim=0.6)}
    assert ("soupe", "soup") in matches
    assert ("dette", "debt") in matches
    # dette/debt is a hidden gem (not orthographically obvious)
    assert matches[("dette", "debt")].cognate is False
    # soupe/soup is a cognate
    assert matches[("soupe", "soup")].cognate is True


def test_hide_cognates(eng):
    gems = eng.golden_matches(min_sim=0.6, include_cognates=False)
    assert all(not m.cognate for m in gems)


def test_meaning_of_with_inflection_fallback(eng):
    assert "envy" in eng.meaning_of("envie")
    # inflected form not in table -> de-inflection falls back to lemma
    assert "envy" in eng.meaning_of("envies")


def test_meaningful_rendering_closes_the_loop(eng):
    rs = eng.meaningful_renderings("envy", top_k=5, min_sound=0.5)
    assert rs, "expected at least one rendering"
    top = rs[0]
    assert top.fr == "envie"
    assert top.via == "direct"
    assert top.meaning == pytest.approx(1.0)


def test_synonyms_bridge(eng):
    # amour -> love, affection ; so love and affection are bridge-synonyms
    syns = eng.synonyms_en("love")
    assert "affection" in syns


def test_homophones_length_window(eng):
    hs = dict((w, s) for w, _, s in eng.homophones_of("soup", min_sim=0.5))
    assert "soupe" in hs


# --------------------------------------------------------------------------- #
# end-to-end on real data (skipped if absent)
# --------------------------------------------------------------------------- #
_DATA = os.path.join(ROOT, "data")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(_DATA, "fr_en_gloss.tsv")),
    reason="real data not present",
)
def test_real_data_has_known_gems():
    eng = MapOfMeaning()
    pairs = {(m.fr, m.en) for m in eng.golden_matches(min_sim=0.85,
                                                       include_cognates=False)}
    assert ("dette", "debt") in pairs
    assert ("estime", "esteem") in pairs
