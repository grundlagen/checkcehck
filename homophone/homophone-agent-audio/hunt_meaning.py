#!/usr/bin/env python3
"""
hunt_meaning.py
~~~~~~~~~~~~~~~

Driver for the *map of meaning* hunt.

It loads the homophone ``pairbank.tsv`` as a bipartite sound-bridge graph,
derives a phonetic-synonym layer from shared neighbours, seeds a meaning oracle
(curated EN<->FR glosses, plus sentence embeddings when installed), hunts
connecting chains out of the glossed anchors, then runs the whittling loop that
condenses the well-supported correspondences into a stable map.

It writes three artifacts under ``--out-dir``:

* ``meaning_map.json``        — full machine-readable result.
* ``meaning_correspondences.tsv`` — the surviving EN<->FR correspondences.
* ``MEANING_REPORT.md``       — a human-readable narrative of the hunt.

Usage::

    python hunt_meaning.py \
        --pairbank data/pairbank.tsv \
        --glosses  data/seed_glosses.tsv \
        --out-dir  out

Everything runs on the standard library; embeddings only sharpen the sense
layer when ``sentence_transformers`` happens to be installed.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

from src.meaning_map import (
    MeaningOracle,
    build_sound_graph,
    Node,
)


def _fmt_path(path: List[Node]) -> str:
    return "  ->  ".join(f"{lang}:{form}" for lang, form in path)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Hunt for the cross-lingual map of meaning.")
    parser.add_argument("--pairbank", default=os.path.join(here, "data", "pairbank.tsv"))
    parser.add_argument("--glosses", default=os.path.join(here, "data", "seed_glosses.tsv"))
    parser.add_argument("--out-dir", default=os.path.join(here, "out"))
    parser.add_argument("--limit", type=int, default=None, help="Cap pairbank rows (quick runs).")
    parser.add_argument("--iterations", type=int, default=6, help="Whittling rounds.")
    parser.add_argument("--top", type=int, default=40, help="Correspondences to surface.")
    parser.add_argument("--max-chains", type=int, default=8, help="Chains kept per anchor.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/5] Loading sound bridges from {args.pairbank} ...")
    oracle = MeaningOracle(gloss_path=args.glosses)
    g = build_sound_graph(args.pairbank, oracle=oracle, limit=args.limit)
    n_nodes = len(g.sound)
    n_edges = sum(len(v) for v in g.sound.values()) // 2
    print(f"      {n_nodes} nodes, {n_edges} sound bridges. Meaning oracle tier: {oracle.tier}")

    print("[2/5] Deriving phonetic-synonym layer (synonyms on top) ...")
    n_syn = g.derive_synonyms()
    print(f"      {n_syn} synonym edges derived.")

    # Anchors = glossed EN/FR forms that actually appear in the sound graph.
    anchors = [node for node in g.sound.keys() if oracle.gloss(node) != node[1]]
    print(f"[3/5] Hunting chains from {len(anchors)} glossed anchors ...")
    chains = g.hunt_chains(anchors, max_chains_per_anchor=args.max_chains)
    chains.sort(key=lambda c: c.score, reverse=True)
    print(f"      {len(chains)} chains found.")

    print(f"[4/5] Whittling over {args.iterations} iterations ...")
    conf, history = g.whittle(chains, iterations=args.iterations)
    print(f"      surviving correspondences per round: {history}")

    print("[5/5] Reporting ...")
    rows = g.report(conf, top=args.top)

    # ---- JSON ----
    grail = [r for r in rows if r["grail"]]
    result = {
        "pairbank": os.path.relpath(args.pairbank, here),
        "oracle_tier": oracle.tier,
        "graph": {"nodes": n_nodes, "sound_bridges": n_edges, "synonym_edges": n_syn},
        "chains_found": len(chains),
        "whittle_history": history,
        "top_chains": [
            {
                "path": [list(p) for p in c.path],
                "sound": round(c.sound_strength, 4),
                "meaning": round(c.meaning_consistency, 4),
                "score": round(c.score, 4),
            }
            for c in chains[:25]
        ],
        "correspondences": rows,
        "grail_count": len(grail),
    }
    json_path = os.path.join(args.out_dir, "meaning_map.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    # ---- TSV ----
    tsv_path = os.path.join(args.out_dir, "meaning_correspondences.tsv")
    with open(tsv_path, "w", encoding="utf-8") as fh:
        fh.write("en\tfr\tconfidence\tmeaning_sim\tgrail\ten_gloss\tfr_gloss\n")
        for r in rows:
            fh.write(
                f"{r['en']}\t{r['fr']}\t{r['confidence']}\t{r['meaning_sim']}"
                f"\t{int(bool(r['grail']))}\t{r['en_gloss']}\t{r['fr_gloss']}\n"
            )

    # ---- Markdown narrative ----
    md_path = os.path.join(args.out_dir, "MEANING_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Map of Meaning — Hunt Report\n\n")
        fh.write(
            f"- **Sound bridges**: {n_edges} (from `{os.path.basename(args.pairbank)}`)\n"
            f"- **Nodes**: {n_nodes}  |  **Synonym edges**: {n_syn}\n"
            f"- **Meaning oracle tier**: `{oracle.tier}`\n"
            f"- **Chains found**: {len(chains)}\n"
            f"- **Whittling** (correspondences surviving per round): "
            f"{' -> '.join(map(str, history))}\n\n"
        )
        fh.write("## Grail matches (sound *and* sense agree)\n\n")
        if grail:
            fh.write("| EN | FR | confidence | meaning |\n|---|---|---|---|\n")
            for r in grail:
                fh.write(f"| {r['en']} | {r['fr']} | {r['confidence']} | {r['fr_gloss']} |\n")
        else:
            fh.write(
                "_None this run — no homophone bridge coincided with a seeded "
                "translation. This is the frontier the next routine widens._\n"
            )
        fh.write("\n## Strongest correspondences\n\n")
        fh.write("| EN | FR | confidence | meaning_sim |\n|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['en']} | {r['fr']} | {r['confidence']} | {r['meaning_sim']} |\n")
        fh.write("\n## Strongest connecting chains\n\n")
        for c in chains[:15]:
            fh.write(
                f"- `{_fmt_path(c.path)}`  "
                f"(sound {c.sound_strength:.3f} x meaning {c.meaning_consistency:.3f} "
                f"= **{c.score:.3f}**)\n"
            )
        fh.write("\n_Sound is data; meaning is the frontier. See `NEXT_ROUTINE.md`._\n")

    print(f"\nWrote:\n  {json_path}\n  {tsv_path}\n  {md_path}")
    print(f"Grail matches (sound+sense): {len(grail)}")


if __name__ == "__main__":
    main()
