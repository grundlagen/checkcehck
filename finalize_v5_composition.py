#!/usr/bin/env python3
"""Finalize dictionary v5 into a composition-ready release.

This is the GPT-reviewed version of the v5 finalization pass. It keeps the
original tier as base_tier, splits B into B_safe/B_reservoir, computes cheap vs
expensive gaps, and uses an effective_gap_ratio for composition usability so
licensed offglide/schwa/h gaps do not over-penalize entries such as dough~dos.

Inputs, by default:
  dictionary-v5.json

Outputs, by default:
  dictionary-v5.json              updated in place unless --no-json-update
  dictionary-v5.tsv               extended TSV
  composition-index.json          lookup indexes for constrained composition

Usage:
  python finalize_v5_composition.py
  python finalize_v5_composition.py --input dictionary-v5.json --tsv dictionary-v5.tsv --index composition-index.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

CHEAP_GAP_SEGS = {"ʊ", "ɪ", "j", "w", "ə", "ɚ", "h"}
FLAG_KEYS = ["multiword", "cognate", "loanword", "pairbank", "decoder"]


def clean_seg(seg: Any) -> str:
    """Normalize a segment for cheap-gap classification."""
    return str(seg).replace("ː", "").replace(":", "").strip()


def coerce_align(e: dict[str, Any]) -> list[tuple[str, str, float]]:
    """Return alignment triples from either JSON align or TSV-style alignment.

    Expected native shape is [[en_seg, fr_seg, cost], ...]. If only the string
    field exists, parse JSON when possible. Bad/missing alignment becomes [].
    """
    align = e.get("align")
    if isinstance(align, list):
        out = []
        for item in align:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                out.append((str(item[0]), str(item[1]), item[2]))
        return out

    raw = e.get("alignment")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            out = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    out.append((str(item[0]), str(item[1]), item[2]))
            return out
    return []


def onset_coda_from_ipa(ipa: str) -> tuple[str, str]:
    """Best-effort onset/coda from a broad IPA string.

    This is a fallback only. If the JSON already has en_onset/en_coda/fr_onset/
    fr_coda, those are preserved. Segment strings in v5 are normally space-free;
    when spaces exist, first/last segment are used.
    """
    toks = [t for t in str(ipa).split() if t]
    if toks:
        return toks[0], toks[-1]
    s = str(ipa).strip()
    if not s:
        return "", ""
    return s[0], s[-1]


def vc_class(seg: str) -> str:
    vowels = set("aeiouyɑɒɔɛæəɜɚɝɪʊøœœ̃ɑ̃ɛ̃ɔ̃ɛ̃̃ɶɨɯɤʌ")
    if not seg:
        return ""
    return "V" if any(ch in vowels for ch in seg) else "C"


def derive(e: dict[str, Any]) -> dict[str, Any]:
    """Derive composition fields for one dictionary entry."""
    e = dict(e)
    align = coerce_align(e)
    n = len(align)

    e["base_tier"] = e.get("base_tier", e.get("tier", ""))

    gaps = [(x, y, c) for x, y, c in align if x == "·" or y == "·"]
    cheap = 0
    for x, y, _c in gaps:
        seg = clean_seg(x if x != "·" else y)
        if seg in CHEAP_GAP_SEGS:
            cheap += 1

    total_gaps = len(gaps)
    expensive = total_gaps - cheap

    e["gap_ratio"] = round(total_gaps / n, 3) if n else 1.0
    e["cheap_gap_ratio"] = round(cheap / n, 3) if n else 1.0
    e["expensive_gap_ratio"] = round(expensive / n, 3) if n else 1.0
    e["effective_gap_ratio"] = round(e["expensive_gap_ratio"] + 0.25 * e["cheap_gap_ratio"], 3)

    es, fs = e.get("en_syll"), e.get("fr_syll")
    if es is not None and fs is not None:
        es_i = int(es)
        fs_i = int(fs)
        e["syllable_delta"] = abs(es_i - fs_i)
        e["huge_deletion"] = (
            (es_i >= 3 and fs_i <= 1)
            or (fs_i >= 3 and es_i <= 1)
            or abs(es_i - fs_i) >= 3
        )
    else:
        e["syllable_delta"] = 99
        e["huge_deletion"] = True

    tier = str(e.get("tier", ""))
    score = float(e.get("score", 0.0))
    if tier == "B":
        b_safe = (
            score >= 0.72
            and e["syllable_delta"] <= 1
            and e["effective_gap_ratio"] <= 0.20
            and not e["huge_deletion"]
        )
        e["tier"] = "B_safe" if b_safe else "B_reservoir"

    t = e.get("tier", "")
    e["usable_for_composition"] = (
        t == "S"
        or (
            t == "A"
            and e["syllable_delta"] <= 1
            and e["effective_gap_ratio"] <= 0.30
            and not e["huge_deletion"]
        )
        or t == "B_safe"
    )

    # Preserve existing junction fields, but backfill if absent.
    en_on, en_co = onset_coda_from_ipa(e.get("en_ipa", ""))
    fr_on, fr_co = onset_coda_from_ipa(e.get("fr_ipa", ""))
    e.setdefault("en_onset", en_on)
    e.setdefault("en_coda", en_co)
    e.setdefault("fr_onset", fr_on)
    e.setdefault("fr_coda", fr_co)
    e["en_onset_class"] = vc_class(e.get("en_onset", ""))
    e["en_coda_class"] = vc_class(e.get("en_coda", ""))
    e["fr_onset_class"] = vc_class(e.get("fr_onset", ""))
    e["fr_coda_class"] = vc_class(e.get("fr_coda", ""))

    if not e.get("alignment"):
        e["alignment"] = json.dumps(align, ensure_ascii=False, separators=(",", ":"))
    if not e.get("align"):
        e["align"] = [[x, y, c] for x, y, c in align]

    return e


def flags_for(e: dict[str, Any]) -> str:
    return ",".join(k for k in FLAG_KEYS if e.get(k))


TSV_COLUMNS = [
    "base_tier", "tier", "score", "direction", "en", "fr", "flags",
    "en_ipa", "fr_ipa", "pivot", "en_syll", "fr_syll",
    "syllable_delta", "gap_ratio", "cheap_gap_ratio", "expensive_gap_ratio",
    "effective_gap_ratio", "huge_deletion", "usable_for_composition",
    "en_onset", "en_coda", "fr_onset", "fr_coda",
    "en_onset_class", "en_coda_class", "fr_onset_class", "fr_coda_class",
    "alignment",
]


def write_tsv(entries: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_COLUMNS)
        for x in entries:
            row = []
            for col in TSV_COLUMNS:
                if col == "flags":
                    row.append(flags_for(x))
                elif col in {"huge_deletion", "usable_for_composition"}:
                    row.append(int(bool(x.get(col))))
                else:
                    row.append(x.get(col, ""))
            writer.writerow(row)


def build_index(entries: list[dict[str, Any]]) -> dict[str, dict[str, list[int]]]:
    idx: dict[str, defaultdict[str, list[int]]] = {
        "pivot": defaultdict(list),
        "first_class": defaultdict(list),
        "final_class": defaultdict(list),
        "en_syll": defaultdict(list),
        "fr_syll": defaultdict(list),
        "syll_pair": defaultdict(list),
        "direction": defaultdict(list),
        "base_tier": defaultdict(list),
        "tier": defaultdict(list),
        "usable": defaultdict(list),
    }
    for i, x in enumerate(entries):
        p = str(x.get("pivot", ""))
        idx["pivot"][p].append(i)
        if p:
            idx["first_class"][p[0]].append(i)
            idx["final_class"][p[-1]].append(i)
        idx["en_syll"][str(x.get("en_syll", ""))].append(i)
        idx["fr_syll"][str(x.get("fr_syll", ""))].append(i)
        idx["syll_pair"][f"{x.get('en_syll','')}:{x.get('fr_syll','')}"].append(i)
        idx["direction"][str(x.get("direction", "en_fr"))].append(i)
        idx["base_tier"][str(x.get("base_tier", ""))].append(i)
        idx["tier"][str(x.get("tier", ""))].append(i)
        idx["usable"][str(int(bool(x.get("usable_for_composition"))))].append(i)
    return {k: dict(v) for k, v in idx.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Finalize v5 dictionary for composition.")
    ap.add_argument("--input", default="dictionary-v5.json", help="Input JSON dictionary")
    ap.add_argument("--json-out", default="dictionary-v5.json", help="Output JSON path")
    ap.add_argument("--tsv", default="dictionary-v5.tsv", help="Extended TSV output")
    ap.add_argument("--index", default="composition-index.json", help="Composition index JSON output")
    ap.add_argument("--no-json-update", action="store_true", help="Do not write updated JSON")
    args = ap.parse_args()

    input_path = Path(args.input)
    entries_raw = json.load(input_path.open("r", encoding="utf-8"))
    if not isinstance(entries_raw, list):
        raise TypeError(f"Expected list in {input_path}, got {type(entries_raw).__name__}")

    entries = [derive(x) for x in entries_raw]

    if not args.no_json_update:
        Path(args.json_out).write_text(json.dumps(entries, ensure_ascii=False, indent=0), encoding="utf-8")

    write_tsv(entries, Path(args.tsv))
    Path(args.index).write_text(json.dumps(build_index(entries), ensure_ascii=False), encoding="utf-8")

    tiers = Counter(x["tier"] for x in entries)
    base_tiers = Counter(x.get("base_tier", "") for x in entries)
    usable = sum(1 for x in entries if x.get("usable_for_composition"))
    print(f"base_tiers: {dict(base_tiers)}")
    print(f"tiers: {dict(tiers)}")
    print(f"usable_for_composition: {usable}/{len(entries)}")
    print(f"wrote {args.tsv}, {args.index}" + ("" if not args.no_json_update else " (JSON not updated)"))


if __name__ == "__main__":
    main()
