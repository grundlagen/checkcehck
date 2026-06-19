"""
semantic_chain.py
~~~~~~~~~~~~~~~~~

The hunt for a **map of meaning**.

This is the engine that the project has been building toward: it walks the two
pair banks (homophone EN->FR, meaning FR->EN) and the synonym bank, and looks
for places where the *sound* path and the *meaning* path agree.

The chain
---------
For one homophone edge ``EN_src ~sound~ FR_tgt`` we build::

    EN_src  ──≈sound≈──▶  FR_tgt  ──=means=──▶  {EN_gloss₁, EN_gloss₂, …}
       │                                              │
       └──────────── do these meet? ──────────────────┘
                    (directly, or via synonyms)

* If ``EN_src`` (or one of its synonyms) is among the French word's English
  glosses, the bridge is **resonant**: it both *sounds* like the source and
  *means* something the source already knows.  These are the gold the whole
  project is mining for — cross-lingual sound-alikes that are also semantically
  honest.
* If they don't meet directly, we still score the *latent* meeting via embedding
  / bag-of-words similarity, so near-misses surface for the next round.

The whittle
-----------
"Whittle down the same meanings over time": we iterate.  Each round keeps the
strongest bridges, then *merges* bridges that land on the same English concept
(via synonyms) into meaning-clusters, deduplicating the survivors.  The set
shrinks and stabilizes — convergence toward a clean map where each meaning has
its best cross-lingual sound-alikes attached.

Everything is pure-Python and degrades gracefully (no required heavy deps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple
import math

from .meaning_bank import MeaningBank, SynonymBank, normalize

# Phonetic similarity (feature-weighted IPA edit distance). Always available.
try:
    from .phone_metric import similarity as _phone_similarity
except Exception:  # pragma: no cover
    _phone_similarity = None  # type: ignore

# IPA conversion (phonemizer if present, else lexique dictionary fallback).
try:
    from .orchestrator import g2p as _g2p
except Exception:  # pragma: no cover
    _g2p = None  # type: ignore

# Latent semantic similarity (sentence-transformers if present, else BoW).
try:
    from .embedding import semantic_similarity as _semantic_similarity
except Exception:  # pragma: no cover
    def _semantic_similarity(a: str, b: str) -> float:  # type: ignore
        a_set, b_set = set(a.lower().split()), set(b.lower().split())
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / len(a_set | b_set)


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass
class Bridge:
    """One walked chain: an EN source, a FR homophone, and their meeting."""

    en_src: str
    fr_tgt: str
    phon: float                     # sound resonance in [0,1]
    glosses: Tuple[str, ...]        # English meanings of the French word
    meaning: float                  # meaning resonance in [0,1]
    resonance: float                # combined score
    kind: str                       # "resonant" | "latent" | "unglossed"
    meeting: Tuple[str, str]        # (en-side concept, fr-gloss concept) that met
    path: Tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> Dict[str, object]:
        return {
            "en_src": self.en_src,
            "fr_tgt": self.fr_tgt,
            "phon": round(self.phon, 4),
            "meaning": round(self.meaning, 4),
            "resonance": round(self.resonance, 4),
            "kind": self.kind,
            "glosses": list(self.glosses),
            "meeting": list(self.meeting),
            "path": list(self.path),
        }


# --------------------------------------------------------------------------- #
# Phonetic resonance
# --------------------------------------------------------------------------- #
_IPA_CACHE: Dict[Tuple[str, str], str] = {}


def _ipa(text: str, lang: str) -> str:
    key = (text, lang)
    if key in _IPA_CACHE:
        return _IPA_CACHE[key]
    val = ""
    if _g2p is not None:
        try:
            val = _g2p(text, lang)
        except Exception:
            val = ""
    _IPA_CACHE[key] = val
    return val


def sound_resonance(en_src: str, fr_tgt: str) -> float:
    """Phonetic similarity of the EN source and FR target in [0,1].

    Falls back to a character bigram Dice coefficient when IPA or the phone
    metric is unavailable, so the engine still ranks sensibly with no deps.
    """
    ipa_en = _ipa(en_src, "en")
    ipa_fr = _ipa(fr_tgt, "fr")
    if _phone_similarity is not None and ipa_en and ipa_fr:
        try:
            return float(_phone_similarity(ipa_en, ipa_fr))
        except Exception:
            pass
    return _dice_bigrams(normalize(en_src), normalize(fr_tgt))


def _dice_bigrams(a: str, b: str) -> float:
    a = a.replace(" ", "")
    b = b.replace(" ", "")
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ga = {a[i : i + 2] for i in range(len(a) - 1)} or {a}
    gb = {b[i : i + 2] for i in range(len(b) - 1)} or {b}
    inter = len(ga & gb)
    return 2.0 * inter / (len(ga) + len(gb))


# --------------------------------------------------------------------------- #
# Meaning resonance
# --------------------------------------------------------------------------- #
def meaning_resonance(
    en_src: str,
    glosses: Set[str],
    syns: SynonymBank,
) -> Tuple[float, str, Tuple[str, str]]:
    """Best meeting between the EN source neighbourhood and the FR glosses.

    Returns ``(score, kind, meeting)`` where ``kind`` is:
      * ``"resonant"`` — direct or synonym-mediated concept match (score >= 0.8)
      * ``"latent"``   — only an embedding/overlap similarity was found
      * ``"unglossed"``— the French word has no known meaning yet
    """
    if not glosses:
        return 0.0, "unglossed", (normalize(en_src), "")

    src_neigh = syns.expand(en_src)
    # Expand each gloss into its own synonym neighbourhood once.
    gloss_neighs = {g: syns.expand(g) for g in glosses}

    # 1) Direct lemma identity — the source word *is* one of the meanings.
    for g in glosses:
        if normalize(en_src) == normalize(g):
            return 1.0, "resonant", (normalize(en_src), g)

    # 2) Synonym-mediated meeting.
    best_syn = 0.0
    best_syn_meet: Tuple[str, str] = (normalize(en_src), "")
    for g, gneigh in gloss_neighs.items():
        if src_neigh & gneigh:
            # overlap exists; richer overlap => slightly higher score
            overlap = len(src_neigh & gneigh)
            score = min(0.95, 0.8 + 0.03 * (overlap - 1))
            if score > best_syn:
                best_syn = score
                shared = sorted(src_neigh & gneigh)[0]
                best_syn_meet = (shared, g)
    if best_syn > 0.0:
        return best_syn, "resonant", best_syn_meet

    # 3) Latent similarity — embedding (or BoW) between source and gloss text.
    best_lat = 0.0
    best_lat_meet: Tuple[str, str] = (normalize(en_src), "")
    gloss_blob = ", ".join(sorted(glosses))
    for g in glosses:
        s = _semantic_similarity(en_src, g)
        if s > best_lat:
            best_lat = s
            best_lat_meet = (normalize(en_src), g)
    # also compare against the whole gloss blob (captures multi-word senses)
    s_blob = _semantic_similarity(en_src, gloss_blob)
    if s_blob > best_lat:
        best_lat = s_blob
        best_lat_meet = (normalize(en_src), gloss_blob)
    return best_lat, "latent", best_lat_meet


# --------------------------------------------------------------------------- #
# Bridge construction
# --------------------------------------------------------------------------- #
def build_bridge(
    en_src: str,
    fr_tgt: str,
    meanings: MeaningBank,
    syns: SynonymBank,
    w_sound: float = 0.5,
    w_meaning: float = 0.5,
) -> Bridge:
    phon = sound_resonance(en_src, fr_tgt)
    glosses = meanings.gloss(fr_tgt)
    meaning, kind, meeting = meaning_resonance(en_src, glosses, syns)
    resonance = w_sound * phon + w_meaning * meaning
    path = (
        f"EN:{normalize(en_src)}",
        "≈sound≈",
        f"FR:{normalize(fr_tgt)}",
        "=means=",
        f"EN:{meeting[1]}" if meeting[1] else "EN:?",
    )
    return Bridge(
        en_src=normalize(en_src),
        fr_tgt=normalize(fr_tgt),
        phon=phon,
        glosses=tuple(sorted(glosses)),
        meaning=meaning,
        resonance=resonance,
        kind=kind,
        meeting=meeting,
        path=path,
    )


def build_bridges(
    edges: Sequence[Tuple[str, str]],
    meanings: MeaningBank,
    syns: SynonymBank,
    *,
    w_sound: float = 0.5,
    w_meaning: float = 0.5,
    require_meaning: bool = True,
    min_phon: float = 0.0,
) -> List[Bridge]:
    """Walk every homophone edge into a scored :class:`Bridge`.

    ``edges`` is a sequence of ``(en_src, fr_tgt)`` pairs (e.g. from the
    homophone pairbank).  With ``require_meaning`` we keep only bridges whose
    French target has a known gloss — the part of the map we can actually
    reason about today.
    """
    bridges: List[Bridge] = []
    for en_src, fr_tgt in edges:
        if require_meaning and not meanings.gloss(fr_tgt):
            continue
        b = build_bridge(en_src, fr_tgt, meanings, syns, w_sound, w_meaning)
        if b.phon < min_phon:
            continue
        bridges.append(b)
    bridges.sort(key=lambda x: x.resonance, reverse=True)
    return bridges


# --------------------------------------------------------------------------- #
# The whittle: converge toward a clean map of meaning
# --------------------------------------------------------------------------- #
@dataclass
class WhittleRound:
    index: int
    kept: int
    clusters: int
    mean_resonance: float


def _concept_key(b: Bridge, syns: SynonymBank) -> str:
    """A canonical English concept for clustering equivalent meanings.

    We use the lexicographically smallest member of the meeting concept's
    synonym neighbourhood, so ``aunt``/``auntie`` collapse to one cluster.
    """
    concept = b.meeting[1] or (b.glosses[0] if b.glosses else b.en_src)
    neigh = syns.expand(concept)
    return sorted(neigh)[0] if neigh else normalize(concept)


def whittle(
    bridges: Sequence[Bridge],
    syns: SynonymBank,
    *,
    rounds: int = 4,
    keep_frac: float = 0.7,
    min_keep: int = 1,
) -> Tuple[List[Bridge], List[WhittleRound]]:
    """Iteratively prune and cluster bridges until the map stabilizes.

    Each round:
      1. sort by resonance and keep the top ``keep_frac`` (never below ``min_keep``);
      2. group survivors by canonical English concept;
      3. within each concept keep only the single best bridge (the cleanest
         sound-alike for that meaning).
    Returns the converged bridge list plus a per-round convergence history so
    callers can *see* the whittling happen.
    """
    current = sorted(bridges, key=lambda x: x.resonance, reverse=True)
    history: List[WhittleRound] = []
    for r in range(rounds):
        cut = max(min_keep, int(math.ceil(len(current) * keep_frac)))
        current = current[:cut]

        # Cluster by concept, keep best per concept.
        best_by_concept: Dict[str, Bridge] = {}
        for b in current:
            key = _concept_key(b, syns)
            cur = best_by_concept.get(key)
            if cur is None or b.resonance > cur.resonance:
                best_by_concept[key] = b
        clustered = sorted(
            best_by_concept.values(), key=lambda x: x.resonance, reverse=True
        )

        mean_res = (
            sum(b.resonance for b in clustered) / len(clustered) if clustered else 0.0
        )
        history.append(
            WhittleRound(
                index=r,
                kept=len(clustered),
                clusters=len(best_by_concept),
                mean_resonance=round(mean_res, 4),
            )
        )

        # Converged: clustering no longer removes anything.
        if len(clustered) == len(current):
            current = clustered
            break
        current = clustered

    return current, history


# --------------------------------------------------------------------------- #
# Map of meaning
# --------------------------------------------------------------------------- #
def meaning_map(
    bridges: Sequence[Bridge],
    syns: SynonymBank,
) -> Dict[str, List[Bridge]]:
    """Group bridges by English concept -> its cross-lingual sound-alikes.

    This *is* the map: for each meaning, the French words that both *sound*
    English-ish and *mean* the concept, ranked by resonance.
    """
    out: Dict[str, List[Bridge]] = {}
    for b in bridges:
        key = _concept_key(b, syns)
        out.setdefault(key, []).append(b)
    for key in out:
        out[key].sort(key=lambda x: x.resonance, reverse=True)
    return dict(sorted(out.items(), key=lambda kv: -max(b.resonance for b in kv[1])))
