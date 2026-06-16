"""
semantic_oracle.py — pluggable judge of *meaning* overlap between an English
word and a French word.

The Map of Meaning needs two relations over the same nodes:
  * sound   (homophony)  — supplied by the IPA lexica + phon.py
  * meaning (translation) — supplied by an oracle here

An oracle returns ``SemanticVerdict(score in [0,1], basis)`` where ``basis``
explains *how* the link was found:
  * ``translation``    direct entry in the bilingual dictionary
  * ``synonym-chain``  the homophone landed on a synonym of a true translation
  * ``unknown``        no meaning signal (score 0) — a sound coincidence only

Default backend: ``SeedDictOracle`` (offline, curated seed).  A network/LLM
backend can be dropped in by implementing ``score(en, fr) -> SemanticVerdict``.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import seed_bilingual


@dataclass
class SemanticVerdict:
    score: float
    basis: str  # translation | synonym-chain | unknown


class SemanticOracle:
    def score(self, en_word: str, fr_word: str) -> SemanticVerdict:  # pragma: no cover
        raise NotImplementedError


class SeedDictOracle(SemanticOracle):
    """Meaning judge backed by the curated seed dictionary + FR synonyms."""

    def __init__(self) -> None:
        self.en_fr = seed_bilingual.EN_FR
        self.fr_syn = seed_bilingual.FR_SYNONYMS
        # also index "to X" verbs under bare "X"
        self._bare = {}
        for en, frs in self.en_fr.items():
            key = en[3:] if en.startswith("to ") else en
            self._bare.setdefault(key, set()).update(frs)

    def _translations(self, en_word: str) -> set[str]:
        w = en_word.lower()
        out = set(self.en_fr.get(w, set()))
        out |= self._bare.get(w, set())
        return out

    def score(self, en_word: str, fr_word: str) -> SemanticVerdict:
        fr = fr_word.lower()
        trans = self._translations(en_word)
        if not trans:
            return SemanticVerdict(0.0, "unknown")
        if fr in trans:
            return SemanticVerdict(1.0, "translation")
        # one-hop synonym chain: fr is a synonym of a true translation,
        # or a true translation is a synonym of fr
        syn_of_fr = self.fr_syn.get(fr, set())
        if trans & syn_of_fr:
            return SemanticVerdict(0.75, "synonym-chain")
        for t in trans:
            if fr in self.fr_syn.get(t, set()):
                return SemanticVerdict(0.75, "synonym-chain")
        return SemanticVerdict(0.0, "unknown")
