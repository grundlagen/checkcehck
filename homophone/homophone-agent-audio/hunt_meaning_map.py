#!/usr/bin/env python3
"""
hunt_meaning_map.py
~~~~~~~~~~~~~~~~~~~

Hunt for a *map of meaning* across the two pair banks.

Pipeline
--------
1. Load the **homophone bank** (EN ~sound~ FR) from ``data/pairbank.tsv``.
2. Load the **meaning bank** (FR -> EN glosses) and the **synonym bank**.
3. Walk every homophone edge into a scored :class:`~src.semantic_chain.Bridge`,
   measuring sound resonance *and* meaning resonance.
4. **Whittle**: iteratively prune + cluster until the map stabilizes.
5. Emit ``map_of_meaning.json`` and a human-readable report to stdout.

Run::

    python hunt_meaning_map.py --top 30
    python hunt_meaning_map.py --w-sound 0.4 --w-meaning 0.6 --rounds 5

No third-party dependencies are required; the engine degrades gracefully.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

# Allow running as a script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.meaning_bank import MeaningBank, SynonymBank, normalize  # noqa: E402
from src.semantic_chain import (  # noqa: E402
    build_bridges,
    whittle,
    meaning_map,
)


def load_homophone_edges(
    path: str,
    direction: Tuple[str, str] = ("en", "fr"),
    tags: Tuple[str, ...] = ("homophone", "phrase"),
) -> List[Tuple[str, str]]:
    """Read (en_src, fr_tgt) pairs from the homophone pairbank TSV."""
    edges: List[Tuple[str, str]] = []
    src_dir, tgt_dir = direction
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}

        def col(cols: List[str], name: str) -> str:
            pos = idx.get(name)
            if pos is None or pos >= len(cols):
                return ""
            return cols[pos]

        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if (col(cols, "src_lang"), col(cols, "tgt_lang")) != (src_dir, tgt_dir):
                continue
            if tags and col(cols, "tag") not in tags:
                continue
            src = normalize(col(cols, "src"))
            tgt = normalize(col(cols, "tgt"))
            if src and tgt:
                edges.append((src, tgt))
    return edges


def find_pairbank() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.path.join(here, "data", "pairbank.tsv"),
        os.path.join(here, "data_small", "pairbank.tsv"),
    ):
        if os.path.exists(cand):
            return cand
    raise SystemExit("Could not find data/pairbank.tsv")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hunt a cross-lingual map of meaning.")
    ap.add_argument("--pairbank", default=None, help="Path to homophone pairbank TSV.")
    ap.add_argument("--w-sound", type=float, default=0.5, help="Weight on sound resonance.")
    ap.add_argument("--w-meaning", type=float, default=0.5, help="Weight on meaning resonance.")
    ap.add_argument("--min-phon", type=float, default=0.0, help="Drop bridges below this sound score.")
    ap.add_argument("--rounds", type=int, default=4, help="Whittle rounds.")
    ap.add_argument("--keep-frac", type=float, default=0.7, help="Fraction kept each whittle round.")
    ap.add_argument("--top", type=int, default=25, help="How many bridges to print.")
    ap.add_argument("--all-edges", action="store_true",
                    help="Keep edges even when the FR target has no known gloss.")
    ap.add_argument("--out", default="map_of_meaning.json", help="Output JSON path.")
    args = ap.parse_args(argv)

    pairbank = args.pairbank or find_pairbank()
    edges = load_homophone_edges(pairbank)
    meanings = MeaningBank.load()
    syns = SynonymBank.load()

    print(f"# Homophonic–Semantic Map of Meaning Hunt", file=sys.stderr)
    print(f"  homophone edges : {len(edges):,}", file=sys.stderr)
    print(f"  meaning bank    : {len(meanings):,} French entries", file=sys.stderr)
    print(f"  synonym bank    : {len(syns.syn):,} English entries"
          f"{' (+WordNet)' if syns._wordnet_ok else ''}", file=sys.stderr)

    bridges = build_bridges(
        edges,
        meanings,
        syns,
        w_sound=args.w_sound,
        w_meaning=args.w_meaning,
        require_meaning=not args.all_edges,
        min_phon=args.min_phon,
    )
    print(f"  bridges built   : {len(bridges):,}", file=sys.stderr)

    kept, history = whittle(
        bridges, syns, rounds=args.rounds, keep_frac=args.keep_frac
    )

    resonant = [b for b in bridges if b.kind == "resonant"]
    print(f"  resonant bridges: {len(resonant):,} "
          f"(sound AND meaning agree)\n", file=sys.stderr)

    # ----- human-readable report -----
    print("=" * 72)
    print("RESONANT BRIDGES — cross-lingual sound-alikes that also tell the truth")
    print("=" * 72)
    for b in sorted(resonant, key=lambda x: x.resonance, reverse=True)[: args.top]:
        glosses = ", ".join(b.glosses)
        print(f"  {b.resonance:.3f}  EN '{b.en_src}'  ≈  FR '{b.fr_tgt}'  "
              f"→ means [{glosses}]")
        print(f"          sound={b.phon:.3f}  meaning={b.meaning:.3f}  "
              f"meet={b.meeting[0]}↔{b.meeting[1]}")

    print("\n" + "=" * 72)
    print("WHITTLE — convergence toward one sound-alike per meaning")
    print("=" * 72)
    print(f"  start: {len(bridges)} bridges")
    for h in history:
        print(f"  round {h.index}: kept {h.kept:>4}  "
              f"concept-clusters {h.clusters:>4}  "
              f"mean-resonance {h.mean_resonance:.3f}")

    mmap = meaning_map(kept, syns)
    print("\n" + "=" * 72)
    print("MAP OF MEANING — concept → best cross-lingual sound-alike")
    print("=" * 72)
    for concept, bl in list(mmap.items())[: args.top]:
        top = bl[0]
        print(f"  [{concept}]  ← FR '{top.fr_tgt}' (EN '{top.en_src}', "
              f"resonance {top.resonance:.3f})")

    # ----- machine-readable map -----
    out = {
        "meta": {
            "homophone_edges": len(edges),
            "meaning_entries": len(meanings),
            "synonym_entries": len(syns.syn),
            "wordnet": syns._wordnet_ok,
            "weights": {"sound": args.w_sound, "meaning": args.w_meaning},
            "bridges_built": len(bridges),
            "resonant_bridges": len(resonant),
        },
        "whittle_history": [h.__dict__ for h in history],
        "resonant": [b.as_dict() for b in
                     sorted(resonant, key=lambda x: x.resonance, reverse=True)],
        "map": {
            concept: [b.as_dict() for b in bl]
            for concept, bl in mmap.items()
        },
    }
    out_path = os.path.join(os.path.dirname(pairbank), "..", args.out)
    out_path = os.path.abspath(out_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\nWrote map → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
