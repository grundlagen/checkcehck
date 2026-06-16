"""
resonance.py — the heart of the Map of Meaning.

We overlay two relations on the same FR/EN word nodes:

    sound:   en_word  ~hom~   fr_word     (IPA near-equality, phon.py)
    meaning: en_word  ~mean~  fr_word     (semantic_oracle)

A *resonance* is a pair where both relations hold.  We hunt from both ends:

  1. CONCEPT-ANCHORED ("meaning -> sound").  For every known concept (seed
     EN->FR entry, plus one synonym hop) we already know the meaning link; we
     measure how close the two words *sound*.  This paints the full map: for
     each concept, the cross-lingual phonetic distance.  Where the translation
     is *also* a near-homophone we have GOLD — a word that both sounds like and
     means its counterpart.

  2. SOUND-ANCHORED ("sound -> meaning").  Over the full lexica we find exact
     cross-lingual homophones (normalised-IPA equality) and ask the oracle
     whether they also share meaning.  Confirmed -> reservoir; unexplained ->
     the FRONTIER (false friends / undiscovered links) that future routines
     whittle.

Tiers, by (phon, sem):
    GOLD     phon >= 0.90 and sem >= 0.95   — sound AND meaning converge
    SILVER   phon >= 0.80 and sem >= 0.70
    BRONZE   phon >= 0.65 and sem >= 0.70
    FRONTIER phon >= 0.90 and sem == 0       — strong sound, meaning unknown
    (anything else is dropped as noise)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

from . import phon
from .lexicon_index import Lexicon
from .semantic_oracle import SemanticOracle
from . import journal


@dataclass
class Resonance:
    en_word: str
    en_ipa: str
    fr_word: str
    fr_ipa: str
    phon_sim: float
    sem_score: float
    basis: str          # translation | synonym-chain | unknown
    direction: str      # concept | sound
    tier: str

    def key(self) -> str:
        return journal.pair_key(self.en_word, self.fr_word)


def classify(phon_sim: float, sem_score: float) -> str:
    if phon_sim >= 0.90 and sem_score >= 0.95:
        return "GOLD"
    if phon_sim >= 0.80 and sem_score >= 0.70:
        return "SILVER"
    if phon_sim >= 0.65 and sem_score >= 0.70:
        return "BRONZE"
    if phon_sim >= 0.90 and sem_score == 0.0:
        return "FRONTIER"
    return "NOISE"


def concept_hunt(en_lex: Lexicon, fr_lex: Lexicon,
                 oracle: SemanticOracle) -> List[Resonance]:
    """Meaning is known; measure the sound. Iterates the oracle's vocabulary."""
    out: List[Resonance] = []
    seed = getattr(oracle, "en_fr", {})
    fr_syn = getattr(oracle, "fr_syn", {})
    for en_word, fr_set in seed.items():
        bare = en_word[3:] if en_word.startswith("to ") else en_word
        if bare not in en_lex.by_word:
            continue
        en_ipa = en_lex.by_word[bare]
        # candidate FR forms: direct translations + their synonyms
        fr_candidates = set(fr_set)
        for fr in fr_set:
            fr_candidates |= fr_syn.get(fr, set())
        for fr in fr_candidates:
            if fr not in fr_lex.by_word:
                continue
            fr_ipa = fr_lex.by_word[fr]
            ps = phon.similarity(en_ipa, fr_ipa)
            verdict = oracle.score(en_word, fr)
            if verdict.score == 0.0:
                continue
            tier = classify(ps, verdict.score)
            if tier == "NOISE":
                continue
            out.append(Resonance(bare, en_ipa, fr, fr_ipa, round(ps, 4),
                                  verdict.score, verdict.basis, "concept", tier))
    return out


def sound_hunt(en_lex: Lexicon, fr_lex: Lexicon, oracle: SemanticOracle,
               *, max_per_norm: int = 6) -> List[Resonance]:
    """Sound is exact (normalised-IPA equality); ask whether meaning agrees."""
    out: List[Resonance] = []
    shared = set(en_lex.by_norm) & set(fr_lex.by_norm)
    for norm in shared:
        en_words = sorted(en_lex.by_norm[norm])[:max_per_norm]
        fr_words = sorted(fr_lex.by_norm[norm])[:max_per_norm]
        for ew in en_words:
            en_ipa = en_lex.by_word[ew]
            for fw in fr_words:
                fr_ipa = fr_lex.by_word[fw]
                ps = phon.similarity(en_ipa, fr_ipa)
                if ps < 0.90:
                    continue
                verdict = oracle.score(ew, fw)
                tier = classify(ps, verdict.score)
                if tier == "NOISE":
                    continue
                out.append(Resonance(ew, en_ipa, fw, fr_ipa, round(ps, 4),
                                     verdict.score, verdict.basis, "sound", tier))
    return out


def dedupe(resonances: List[Resonance]) -> List[Resonance]:
    """Keep the strongest record per (en,fr) pair."""
    best: dict[str, Resonance] = {}
    for r in resonances:
        k = r.key()
        cur = best.get(k)
        score = r.phon_sim + r.sem_score
        if cur is None or score > (cur.phon_sim + cur.sem_score):
            best[k] = r
    return list(best.values())


def to_record(r: Resonance, run_id: str) -> dict:
    rec = asdict(r)
    rec["key"] = r.key()
    rec["run_id"] = run_id
    rec["ts"] = journal.now()
    return rec
