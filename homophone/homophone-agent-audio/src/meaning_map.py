"""
meaning_map.py
~~~~~~~~~~~~~~

The *map of meaning* engine.

The rest of this package hunts **homophones**: forms that sound alike across
EN and FR.  The ``pairbank.tsv`` it produces is, at heart, a pure *sound
bridge* graph — each row says "this English shape sounds like that French
word".  It carries no meaning of its own.

This module begins the next leg of the journey: layering **sense** on top of
**sound** so that connecting chains can be walked from a French meaning, over
a homophone bridge, into an English meaning (and back), reinforcing the
correspondences that many independent chains agree on and letting the spurious
ones decay.  That stabilised residue is what we call the *map of meaning*.

The design follows the project's house rules:

* **Honest about grounding.**  Sound is real data (the pairbank).  Meaning is
  the frontier.  A pluggable :class:`MeaningOracle` supplies sense similarity —
  a sentence-embedding model when available, a curated EN<->FR gloss seed when
  present, and an orthographic-cognate proxy as a last resort.  The report
  always states which oracle spoke.
* **Self-contained.**  Everything here runs on the standard library.  Heavy
  models only *improve* the meaning layer; they are never required to run.
* **Synonyms from the bridges themselves.**  Two English forms that both sound
  like the same French word are *phonetic synonyms*; the same holds mirror-wise
  for French.  This "mapping synonyms on top" needs no external thesaurus — it
  falls straight out of the pairbank's shared neighbours.

Pipeline
--------

1. :func:`build_sound_graph` — bipartite EN<->FR sound-bridge graph.
2. :meth:`MeaningGraph.derive_synonyms` — phonetic-synonym edges from shared
   neighbours ("synonyms on top").
3. :meth:`MeaningGraph.hunt_chains` — enumerate sound->synonym->sound chains and
   score each by ``sound_strength * meaning_consistency``.
4. :meth:`MeaningGraph.whittle` — iterative consensus: correspondences with
   broad independent chain support rise; the unsupported decay below threshold
   and are whittled away.  This is the "whittle down the same meanings over
   time" loop made literal.
5. :meth:`MeaningGraph.report` — emit the surviving map as JSON-ready data.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Iterable, Optional, Callable
import csv
import math

# ---------------------------------------------------------------------------
# Node identity
# ---------------------------------------------------------------------------
# A node is a (lang, form) pair.  We keep them as plain tuples for hashing
# speed and memory; helpers below make intent readable at call sites.
Node = Tuple[str, str]


def en(form: str) -> Node:
    return ("en", form)


def fr(form: str) -> Node:
    return ("fr", form)


# ---------------------------------------------------------------------------
# Meaning oracle (pluggable sense layer)
# ---------------------------------------------------------------------------
def _ortho_norm(s: str) -> str:
    """Lowercase, strip non-letters; used by the cognate fallback."""
    import unicodedata
    import re
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z]", "", s.lower())


class MeaningOracle:
    """Supplies cross-form meaning similarity in ``[0, 1]`` and glosses.

    Three tiers, tried in order of trustworthiness:

    * **embedding** — if ``sentence_transformers`` loaded a model, use cosine
      between sentence embeddings (works cross-lingually, if coarsely).
    * **gloss seed** — a curated ``EN<TAB>FR`` translation table; exact
      translations score 1.0 and supply human-readable glosses.
    * **cognate proxy** — orthographic Jaccard, a weak last resort so the
      pipeline still produces *something* with zero resources.

    The active tier is recorded in :attr:`tier` for honest reporting.
    """

    def __init__(self, gloss_path: Optional[str] = None) -> None:
        self.tier = "cognate"
        self._embedder = None
        # EN form -> set of FR translations (and the mirror) from the seed.
        self._en2fr: Dict[str, Set[str]] = defaultdict(set)
        self._fr2en: Dict[str, Set[str]] = defaultdict(set)
        # Best human gloss for a form, for the report.
        self._gloss: Dict[Node, str] = {}

        if gloss_path:
            self._load_glosses(gloss_path)

        # Try the embedding tier last so that, if present, it wins.
        try:
            from .embedding import _EMBEDDER  # type: ignore
            if _EMBEDDER is not None:
                self._embedder = _EMBEDDER
                self.tier = "embedding"
        except Exception:
            self._embedder = None

        if self._embedder is None and (self._en2fr or self._fr2en):
            self.tier = "gloss"

    def _load_glosses(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    e, f = parts[0].strip().lower(), parts[1].strip().lower()
                    gloss = parts[2].strip() if len(parts) > 2 else f
                    if not e or not f:
                        continue
                    self._en2fr[e].add(f)
                    self._fr2en[f].add(e)
                    self._gloss[en(e)] = gloss
                    self._gloss[fr(f)] = gloss
        except FileNotFoundError:
            pass

    def gloss(self, node: Node) -> str:
        return self._gloss.get(node, node[1])

    def are_translations(self, a: Node, b: Node) -> bool:
        """True iff the seed table records ``a`` and ``b`` as translations."""
        (la, fa), (lb, fb) = a, b
        if {la, lb} != {"en", "fr"}:
            return False
        e = fa if la == "en" else fb
        f = fb if lb == "fr" else fa
        return f in self._en2fr.get(e, set())

    def similarity(self, a: Node, b: Node) -> float:
        """Meaning similarity in ``[0, 1]`` between two nodes."""
        if a == b:
            return 1.0
        # Tier 1: curated translation pair -> certain sense identity.
        if self.are_translations(a, b):
            return 1.0
        # Tier 2: embeddings (cross-lingual cosine, mapped to [0,1]).
        if self._embedder is not None:
            try:
                ga, gb = self.gloss(a), self.gloss(b)
                vecs = self._embedder.encode([ga, gb])
                va, vb = vecs[0], vecs[1]
                dot = float(sum(x * y for x, y in zip(va, vb)))
                na = math.sqrt(sum(x * x for x in va))
                nb = math.sqrt(sum(y * y for y in vb))
                if na and nb:
                    return max(0.0, min(1.0, (dot / (na * nb) + 1.0) / 2.0))
            except Exception:
                pass
        # Tier 3: orthographic cognate proxy (weak but never empty-handed).
        na_s, nb_s = set(_ortho_norm(a[1])), set(_ortho_norm(b[1]))
        if not na_s or not nb_s:
            return 0.0
        return len(na_s & nb_s) / len(na_s | nb_s)


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------
@dataclass
class Chain:
    """A connecting chain that transports a meaning across sound bridges."""
    path: List[Node]
    sound_strength: float          # product of per-bridge sound confidences
    meaning_consistency: float     # oracle sim between the two endpoints
    score: float                   # sound_strength * meaning_consistency

    def endpoints(self) -> Tuple[Node, Node]:
        return self.path[0], self.path[-1]


@dataclass
class MeaningGraph:
    """Bipartite EN<->FR sound graph with a derived synonym layer.

    Attributes
    ----------
    sound : adjacency for homophone bridges, ``node -> {neighbour: weight}``.
    syn   : adjacency for phonetic-synonym edges (same language), built by
            :meth:`derive_synonyms`.
    """
    sound: Dict[Node, Dict[Node, float]] = field(default_factory=lambda: defaultdict(dict))
    syn: Dict[Node, Dict[Node, float]] = field(default_factory=lambda: defaultdict(dict))
    oracle: MeaningOracle = field(default_factory=MeaningOracle)

    # -- construction --------------------------------------------------------
    def add_sound_bridge(self, a: Node, b: Node, weight: float = 1.0) -> None:
        # Bridges are undirected; accumulate weight on repeats (multiplicity
        # is evidence of a robust correspondence).
        self.sound[a][b] = self.sound[a].get(b, 0.0) + weight
        self.sound[b][a] = self.sound[b].get(a, 0.0) + weight

    def nodes(self) -> List[Node]:
        return list(self.sound.keys())

    def derive_synonyms(self, max_degree: int = 60, min_overlap: int = 1) -> int:
        """Build phonetic-synonym edges from shared sound neighbours.

        If two EN forms both bridge to the same FR word, they are sound-kin —
        a *phonetic synonym* pair.  Edge weight is the Jaccard overlap of their
        neighbour sets, so forms that share many bridges bind more tightly.

        To stay tractable on hub nodes (a few FR words attract dozens of EN
        shapes) we skip any pivot whose degree exceeds ``max_degree``; such hubs
        carry little discriminative signal anyway.  Returns the edge count.
        """
        added = 0
        for pivot, neigh in self.sound.items():
            kin = list(neigh.keys())
            if len(kin) < 2 or len(kin) > max_degree:
                continue
            # All co-neighbours of the pivot share at least this pivot.
            for i in range(len(kin)):
                for j in range(i + 1, len(kin)):
                    a, b = kin[i], kin[j]
                    if a[0] != b[0]:
                        continue  # synonyms live within one language
                    na = set(self.sound[a].keys())
                    nb = set(self.sound[b].keys())
                    inter = na & nb
                    if len(inter) < min_overlap:
                        continue
                    w = len(inter) / len(na | nb)
                    if w > self.syn[a].get(b, 0.0):
                        self.syn[a][b] = w
                        self.syn[b][a] = w
                        added += 1
        return added

    # -- chain hunting -------------------------------------------------------
    def hunt_chains(
        self,
        anchors: Iterable[Node],
        max_chains_per_anchor: int = 8,
        meaning_floor: float = 0.0,
    ) -> List[Chain]:
        """Enumerate connecting chains seeded at ``anchors``.

        From each anchor we walk ``sound -> synonym -> sound`` (a single
        homophone hop, an optional in-language synonym slide, then back across
        a homophone), collecting endpoints in the *anchor's own language*.  The
        chain "transports" the anchor's sound; we then ask the oracle whether
        its **meaning** survived the trip.  Chains where sound and sense both
        hold are the threads of the map.
        """
        chains: List[Chain] = []
        for anchor in anchors:
            local: List[Chain] = []
            for mid, w1 in self.sound.get(anchor, {}).items():
                # mid is in the other language. Slide along its synonyms, then
                # cross a homophone bridge back to the anchor's language.
                slides = [(mid, 1.0)] + list(self.syn.get(mid, {}).items())
                for mid2, ws in slides:
                    for endp, w2 in self.sound.get(mid2, {}).items():
                        if endp[0] != anchor[0] or endp == anchor:
                            continue
                        sound_strength = _conf(w1) * ws * _conf(w2)
                        mc = self.oracle.similarity(anchor, endp)
                        if mc < meaning_floor:
                            continue
                        path = [anchor, mid] if mid2 == mid else [anchor, mid, mid2]
                        path = path + [endp]
                        local.append(
                            Chain(
                                path=path,
                                sound_strength=sound_strength,
                                meaning_consistency=mc,
                                score=sound_strength * mc,
                            )
                        )
            local.sort(key=lambda c: c.score, reverse=True)
            chains.extend(local[:max_chains_per_anchor])
        return chains

    # -- whittling (iterative consensus) ------------------------------------
    def whittle(
        self,
        chains: List[Chain],
        iterations: int = 6,
        alpha: float = 0.5,
        prune_below: float = 0.04,
        prune_growth: float = 0.6,
    ) -> Tuple[Dict[Tuple[Node, Node], float], List[int]]:
        """Reinforce well-supported correspondences; let the rest decay.

        A *correspondence* is an unordered EN<->FR pair that lies on **any**
        bridge of **any** chain.  We start broad — every bridge a chain ever
        touched is in play — and tighten over rounds:

        ``c <- (1 - alpha) * c + alpha * support``

        where ``support`` is the score of chains crossing that bridge, each
        amplified by the confidence of the *other* bridges on the same chain
        (independent agreement reinforces).  The pruning threshold **grows**
        every round (``prune_below * (1 + r * prune_growth)``), so weakly
        supported correspondences fall away progressively — the map condenses
        toward the meanings many chains agree on.  Returns the final confidences
        and the per-round surviving counts so callers can watch it whittle.
        """
        # Seed confidence from EVERY bridge on every chain (start broad).
        conf: Dict[Tuple[Node, Node], float] = defaultdict(float)
        support_index: Dict[Tuple[Node, Node], List[Tuple[Chain, int]]] = defaultdict(list)
        for ch in chains:
            for i in range(len(ch.path) - 1):
                a, b = ch.path[i], ch.path[i + 1]
                if a[0] == b[0]:
                    continue  # only score cross-lingual (sound) bridges as correspondences
                key = _undirected(a, b)
                conf[key] += ch.score
                support_index[key].append((ch, i))
        _normalise(conf)

        history: List[int] = [len(conf)]
        for r in range(iterations):
            thresh = prune_below * (1.0 + r * prune_growth)
            support: Dict[Tuple[Node, Node], float] = defaultdict(float)
            for key, refs in support_index.items():
                if conf.get(key, 0.0) < thresh:
                    continue
                for ch, i in refs:
                    # Independent agreement: amplify by the confidence of the
                    # chain's *other* cross-lingual bridges.
                    amp = 1.0
                    for j in range(len(ch.path) - 1):
                        if j == i:
                            continue
                        a2, b2 = ch.path[j], ch.path[j + 1]
                        if a2[0] != b2[0]:
                            amp += conf.get(_undirected(a2, b2), 0.0)
                    support[key] += ch.score * amp
            _normalise(support)
            new_conf: Dict[Tuple[Node, Node], float] = {}
            for key, c in conf.items():
                updated = (1.0 - alpha) * c + alpha * support.get(key, 0.0)
                if updated >= thresh:
                    new_conf[key] = updated
            _normalise(new_conf)
            conf = defaultdict(float, new_conf)
            history.append(len(new_conf))
        return dict(conf), history

    # -- reporting -----------------------------------------------------------
    def report(
        self,
        conf: Dict[Tuple[Node, Node], float],
        top: int = 40,
    ) -> List[Dict[str, object]]:
        """Rank surviving correspondences and attach glosses for the report."""
        rows: List[Dict[str, object]] = []
        for (a, b), c in sorted(conf.items(), key=lambda kv: kv[1], reverse=True)[:top]:
            # Orient en -> fr for readability.
            ena, frb = (a, b) if a[0] == "en" else (b, a)
            rows.append(
                {
                    "en": ena[1],
                    "fr": frb[1],
                    "confidence": round(c, 4),
                    "meaning_sim": round(self.oracle.similarity(ena, frb), 3),
                    "en_gloss": self.oracle.gloss(ena),
                    "fr_gloss": self.oracle.gloss(frb),
                    "grail": bool(self.oracle.are_translations(ena, frb)),
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _conf(weight: float) -> float:
    """Map an accumulated bridge weight to a confidence in ``(0, 1]``.

    Multiplicity is good evidence but with diminishing returns, so we squash
    with ``w / (w + 1)`` shifted to keep a single sighting meaningful.
    """
    return (weight + 0.5) / (weight + 1.0)


def _undirected(a: Node, b: Node) -> Tuple[Node, Node]:
    return (a, b) if a <= b else (b, a)


def _normalise(d: Dict[Tuple[Node, Node], float]) -> None:
    """In-place L-infinity normalisation so confidences stay comparable."""
    if not d:
        return
    m = max(d.values())
    if m <= 0:
        return
    for k in list(d.keys()):
        d[k] = d[k] / m


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def build_sound_graph(
    pairbank_path: str,
    oracle: Optional[MeaningOracle] = None,
    direction: Tuple[str, str] = ("en", "fr"),
    limit: Optional[int] = None,
) -> MeaningGraph:
    """Load a :class:`MeaningGraph` from a ``pairbank.tsv`` file.

    Only rows matching ``direction`` (``src_lang``, ``tgt_lang``) are used.
    ``limit`` caps the number of rows for quick experiments.
    """
    g = MeaningGraph(oracle=oracle or MeaningOracle())
    with open(pairbank_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            if (row.get("src_lang"), row.get("tgt_lang")) != direction:
                continue
            src = (row.get("src") or "").strip().lower()
            tgt = (row.get("tgt") or "").strip().lower()
            if not src or not tgt:
                continue
            g.add_sound_bridge((direction[0], src), (direction[1], tgt))
    return g
