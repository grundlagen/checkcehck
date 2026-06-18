"""
map_of_meaning.py
~~~~~~~~~~~~~~~~~~

**The Map of Meaning** — a cross-lingual graph that overlays two layers on the
same set of word nodes:

* the **SOUND layer** — how words sound (EN↔FR near-homophones), measured by a
  normalised, feature-weighted IPA distance; and
* the **MEANING layer** — what words mean (FR→EN glosses, plus EN↔EN synonym
  bridges induced by shared translations).

The long-term goal this serves (see ``MAP_OF_MEANING.md``) is *homophonic +
semantic matching*: phrases that simultaneously **sound alike** and **mean the
same** across languages.  The cleanest, fully-offline realisation of that goal
is the **golden match**: a French word that sounds like its own English
meaning (``soupe`` /sup/ → *soup*; ``estime`` /ɛstim/ → *esteem*; ``envie``
/ɑ̃vi/ → *envy*).  Those are points where the two layers touch.

Everything here is pure standard library so it runs anywhere the lexica and the
gloss table are present.  Heavy resources (embeddings, large bilingual
dictionaries) are intentionally *pluggable* rather than required — see
``semantic_relatedness`` and ``NEXT_ROUTINE.md``.

Data inputs (all under ``data/``):
    lexique_en.tsv     EN surface -> IPA
    lexique.tsv        FR surface -> IPA
    fr_en_gloss.tsv    FR surface -> EN glosses (the semantic layer seed)
    pairbank.tsv       EN -> FR sound-alike pairs (mined elsewhere)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import unicodedata
from difflib import SequenceMatcher

from .ipa_norm import norm_ipa
from . import phone_metric as _pm


def _obvious(a: str, b: str, thresh: float = 0.8) -> bool:
    """True if two surface forms look obviously alike to a reader.

    Uses a diacritic-stripped sequence-similarity ratio, so it flags
    orthographically transparent pairs (``soup``/``soupe``) while letting
    hidden gems that merely share letters (``dette``/``debt``) through.
    """
    def _flat(s: str) -> str:
        s = unicodedata.normalize("NFD", s.lower())
        return "".join(c for c in s if not unicodedata.combining(c))
    return SequenceMatcher(None, _flat(a), _flat(b)).ratio() >= thresh

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.abspath(os.path.join(_HERE, "..", "data"))


def _data(name: str) -> str:
    return os.path.join(os.environ.get("DATA_DIR", _DATA), name)


# --------------------------------------------------------------------------- #
# Loaders (first surface form wins; comments and headers skipped)
# --------------------------------------------------------------------------- #
def load_lexicon(path: str) -> Dict[str, str]:
    """Load a ``surface<TAB>ipa`` lexicon into ``{surface_lower: ipa}``."""
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            w = parts[0].strip().lower()
            if w and w not in out and parts[1].strip():
                out[w] = parts[1].strip()
    return out


def load_glosses(path: str) -> Dict[str, List[str]]:
    """Load ``fr<TAB>en1|en2|...`` glosses into ``{fr_lower: [en, ...]}``."""
    out: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or parts[0] == "fr":
                continue
            fr = parts[0].strip().lower()
            glosses = [g.strip().lower() for g in parts[1].split("|") if g.strip()]
            if fr and glosses:
                out[fr] = glosses
    return out


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GoldenMatch:
    """A FR word that sounds like one of its own EN meanings."""
    fr: str
    fr_ipa: str
    en: str          # the gloss it sounds like
    en_ipa: str
    sound: float     # normalised phonetic similarity in [0, 1]
    cognate: bool    # orthographically obvious (soup/soupe) vs. a hidden gem


@dataclass(frozen=True)
class Rendering:
    """A FR candidate that both sounds like and (maybe) means an EN query."""
    fr: str
    fr_ipa: str
    sound: float          # how much it sounds like the query
    meaning: float        # how much its gloss relates to the query [0, 1]
    glosses: Tuple[str, ...]
    via: str              # "direct" | "synonym" | "orthographic" | "none"
    score: float          # combined objective


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class MapOfMeaning:
    """Holds the lexica + gloss layer and answers cross-lingual queries."""

    def __init__(
        self,
        en_lex: Optional[Dict[str, str]] = None,
        fr_lex: Optional[Dict[str, str]] = None,
        glosses: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.en = en_lex if en_lex is not None else load_lexicon(_data("lexique_en.tsv"))
        self.fr = fr_lex if fr_lex is not None else load_lexicon(_data("lexique.tsv"))
        self.gloss = glosses if glosses is not None else load_glosses(_data("fr_en_gloss.tsv"))
        # Inverted gloss: EN word -> set of FR words that translate to it.
        self.en_to_fr: Dict[str, Set[str]] = {}
        for fr, gs in self.gloss.items():
            for g in gs:
                self.en_to_fr.setdefault(g, set()).add(fr)
        # Length-bucket index over the FR lexicon for fast sound lookup.
        # Keyed by normalised-IPA length (robust to first-phone shifts such as
        # nasal vowels, which a first-char index would mis-bucket).
        self._fr_buckets: Dict[int, List[Tuple[str, str, str]]] = {}
        for w, ipa in self.fr.items():
            nip = norm_ipa(ipa)
            if not nip:
                continue
            self._fr_buckets.setdefault(len(nip), []).append((w, ipa, nip))

    # -- sound -------------------------------------------------------------- #
    @staticmethod
    def sound_sim(ipa_a: str, ipa_b: str) -> float:
        """Normalised feature-weighted phonetic similarity in [0, 1]."""
        return _pm.similarity(norm_ipa(ipa_a), norm_ipa(ipa_b))

    def homophones_of(
        self,
        en_word: str,
        top_k: int = 20,
        min_sim: float = 0.6,
        len_tol: int = 2,
    ) -> List[Tuple[str, str, float]]:
        """FR words that sound like ``en_word``.

        Scans only the FR first-phone bucket of the query (and is further
        gated by a length window) so a single lookup stays fast over the
        ~250k-entry French lexicon.  Returns ``(fr, fr_ipa, sound)`` sorted by
        descending similarity.
        """
        ipa = self.en.get(en_word.lower())
        if not ipa:
            return []
        q = norm_ipa(ipa)
        if not q:
            return []
        out: List[Tuple[str, str, float]] = []
        for L in range(len(q) - len_tol, len(q) + len_tol + 1):
            for w, raw, nip in self._fr_buckets.get(L, ()):
                s = _pm.similarity(q, nip)
                if s >= min_sim:
                    out.append((w, raw, s))
        out.sort(key=lambda x: x[2], reverse=True)
        return out[:top_k]

    # -- meaning ------------------------------------------------------------ #
    def meaning_of(self, fr_word: str) -> List[str]:
        """EN glosses for a FR word, with a light inflectional fallback."""
        w = fr_word.lower()
        if w in self.gloss:
            return self.gloss[w]
        # crude de-inflection: try common French endings -> lemma.
        for suf, repl in (("s", ""), ("es", ""), ("e", ""), ("ent", ""),
                          ("nt", ""), ("ais", ""), ("ait", "")):
            if w.endswith(suf) and len(w) - len(suf) >= 2:
                cand = w[: len(w) - len(suf)] + repl
                if cand in self.gloss:
                    return self.gloss[cand]
        return []

    def synonyms_en(self, en_word: str) -> Set[str]:
        """EN words that share a French translation with ``en_word``.

        This is the "synonyms on top" layer: ``en --gloss⁻¹--> fr --gloss-->
        en'`` induces a translation-bridge synonym set, the connective tissue
        for whittling chains down to a shared meaning.
        """
        w = en_word.lower()
        syns: Set[str] = set()
        for fr in self.en_to_fr.get(w, ()):  # FR words meaning w
            for g in self.gloss.get(fr, ()):
                if g != w:
                    syns.add(g)
        return syns

    def semantic_relatedness(self, a: str, b: str) -> Tuple[float, str]:
        """Cheap, dependency-free EN↔EN relatedness with provenance.

        Returns ``(score, via)``.  This is the *pluggable* seam: a future
        routine can swap in embeddings or WordNet behind the same signature
        (see NEXT_ROUTINE.md).  The stdlib version recognises identity,
        translation-bridge synonymy, and orthographic overlap.
        """
        a, b = a.lower().strip(), b.lower().strip()
        if not a or not b:
            return 0.0, "none"
        if a == b:
            return 1.0, "direct"
        if b in self.synonyms_en(a) or a in self.synonyms_en(b):
            return 0.7, "synonym"
        # orthographic Jaccard over character sets as a last resort.
        A, B = set(a), set(b)
        j = len(A & B) / len(A | B) if (A and B) else 0.0
        return (j * 0.4, "orthographic") if j > 0 else (0.0, "none")

    # -- the grand-goal query ---------------------------------------------- #
    def meaningful_renderings(
        self,
        en_word: str,
        top_k: int = 15,
        min_sound: float = 0.55,
        w_sound: float = 0.6,
        w_meaning: float = 0.4,
    ) -> List[Rendering]:
        """FR words that BOTH sound like and relate in meaning to ``en_word``.

        This is the map-of-meaning objective in miniature: walk the sound
        layer out to French neighbours, then pull each candidate's meaning
        back through the gloss layer and score how close it loops to the
        origin.  ``score = w_sound·sound + w_meaning·meaning``.
        """
        out: List[Rendering] = []
        for fr, fr_ipa, sound in self.homophones_of(en_word, top_k=200, min_sim=min_sound):
            glosses = self.meaning_of(fr)
            best_m, via = 0.0, "none"
            for g in glosses:
                m, v = self.semantic_relatedness(en_word, g)
                if m > best_m:
                    best_m, via = m, v
            score = w_sound * sound + w_meaning * best_m
            out.append(Rendering(fr, fr_ipa, sound, best_m, tuple(glosses), via, score))
        out.sort(key=lambda r: r.score, reverse=True)
        return out[:top_k]

    # -- golden matches ----------------------------------------------------- #
    def golden_matches(
        self,
        min_sim: float = 0.6,
        include_cognates: bool = True,
    ) -> List[GoldenMatch]:
        """All FR words that sound like one of their own EN meanings.

        Each result is a point where the sound layer and the meaning layer
        coincide — a self-verifying cross-lingual homophone-synonym.  Set
        ``include_cognates=False`` to hide orthographically obvious pairs
        (soup/soupe) and surface the hidden gems.
        """
        out: List[GoldenMatch] = []
        for fr, glosses in self.gloss.items():
            fr_ipa = self.fr.get(fr)
            if not fr_ipa:
                continue
            for g in glosses:
                en_ipa = self.en.get(g)
                if not en_ipa:
                    continue
                s = self.sound_sim(fr_ipa, en_ipa)
                if s < min_sim:
                    continue
                cog = _obvious(fr, g)
                if cog and not include_cognates:
                    continue
                out.append(GoldenMatch(fr, fr_ipa, g, en_ipa, round(s, 4), cog))
        out.sort(key=lambda m: m.sound, reverse=True)
        return out

    # -- graph export ------------------------------------------------------- #
    def stats(self) -> Dict[str, int]:
        return {
            "en_words": len(self.en),
            "fr_words": len(self.fr),
            "fr_glossed": len(self.gloss),
            "en_glosses": len(self.en_to_fr),
        }
