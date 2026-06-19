#!/usr/bin/env python3
"""
Lightweight, dependency-free self-test for the semantic-chain engine.

Run directly::

    python test_semantic_chain.py

Exits non-zero on the first failed assertion.  Kept plain (no pytest) so it
runs in the minimal environment the rest of this skeleton targets.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.meaning_bank import MeaningBank, SynonymBank
from src.semantic_chain import (
    build_bridge,
    build_bridges,
    meaning_resonance,
    sound_resonance,
    whittle,
    meaning_map,
)


def _check(name: str, cond: bool) -> None:
    status = "ok " if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        raise SystemExit(f"assertion failed: {name}")


def main() -> int:
    meanings = MeaningBank.load()
    syns = SynonymBank.load()

    print("banks loaded:")
    _check("meaning bank non-empty", len(meanings) > 0)
    _check("synonym bank non-empty", len(syns.syn) > 0)

    print("sound resonance:")
    _check("identical forms score high", sound_resonance("net", "net") > 0.9)
    _check("scores are bounded [0,1]", 0.0 <= sound_resonance("tess", "tantes") <= 1.0)

    print("meaning resonance:")
    # 'net' (en) is itself one of the glosses of FR 'net' -> direct identity.
    score, kind, meet = meaning_resonance("net", meanings.gloss("net"), syns)
    _check("direct identity is resonant=1.0", kind == "resonant" and score == 1.0)
    # synonym-mediated: 'memo' is a synonym of 'note'; FR 'note' glosses include 'memo'
    score2, kind2, _ = meaning_resonance("memo", meanings.gloss("note"), syns)
    _check("synonym-mediated meeting is resonant", kind2 == "resonant" and score2 >= 0.8)
    # unknown French word -> unglossed
    score3, kind3, _ = meaning_resonance("net", meanings.gloss("zzzzzz"), syns)
    _check("unknown gloss is unglossed/0.0", kind3 == "unglossed" and score3 == 0.0)

    print("bridge construction:")
    b = build_bridge("net", "net", meanings, syns)
    _check("net↔net bridge is fully resonant", b.kind == "resonant" and b.resonance == 1.0)
    _check("bridge records a path", len(b.path) == 5)

    print("whittle convergence:")
    edges = [("net", "net"), ("memo", "note"), ("aunt", "tante"),
             ("tess", "tantes"), ("ba", "banc"), ("sa", "sang")]
    bridges = build_bridges(edges, meanings, syns, require_meaning=True)
    _check("bridges built from edges", len(bridges) >= 3)
    kept, history = whittle(bridges, syns, rounds=4)
    _check("whittle produced history", len(history) >= 1)
    _check("whittle never grows the set", all(
        history[i].kept >= history[i + 1].kept for i in range(len(history) - 1)
    ))
    _check("kept set is non-empty", len(kept) >= 1)

    print("map of meaning:")
    mmap = meaning_map(kept, syns)
    _check("map has concepts", len(mmap) >= 1)
    _check("net concept present", any(k == "net" or "net" in k for k in mmap))

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
