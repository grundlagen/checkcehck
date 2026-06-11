#!/usr/bin/env python3
"""Finalize dictionary v5 from an existing TSV file.

This runner exists because some v5 exports are available only as TSV with an
`alignment` column like `s:s ɛ:ɛ d:d`, rather than as dictionary-v5.json with
an `align` array. It applies the same composition-readiness logic as
finalize_v5_composition.py:

- preserve original tier as base_tier;
- split B into B_safe / B_reservoir;
- compute cheap / expensive / effective gap ratios;
- use effective_gap_ratio for composition usability;
- backfill onset/coda junction fields and V/C classes from alignment;
- emit an extended TSV and composition-index JSON.

Usage:
  python finalize_v5_from_tsv.py --input dictionaryv5.tsv \
    --tsv dictionary-v5-finalized.tsv \
    --index composition-index-finalized.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

CHEAP_GAP_SEGS = {"ʊ", "ɪ", "j", "w", "ə", "ɚ", "h"}
VOWELS = set("aeiouyɑɒɔɛæəɜɚɝɪʊøœɶɨɯɤʌ̃")

TSV_COLUMNS = [
    "base_tier", "tier", "score", "direction", "en", "fr", "flags",
    "en_ipa", "fr_ipa", "pivot", "en_syll", "fr_syll",
    "syllable_delta", "gap_ratio", "cheap_gap_ratio", "expensive_gap_ratio",
    "effective_gap_ratio", "huge_deletion", "usable_for_composition",
    "en_onset", "en_coda", "fr_onset", "fr_coda",
    "en_onset_class", "en_coda_class", "fr_onset_class", "fr_coda_class",
    "alignment",
]


def is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def sget(row: dict[str, Any], key: str, default: str = "") -> str:
    val = row.get(key, default)
    if is_missing(val):
        return default
    return str(val)


def clean_seg(seg: Any) -> str:
    return str(seg).replace("ː", "").replace(":", "").strip()


def parse_alignment(raw: Any) -> list[tuple[str, str, float]]:
    """Parse v5 TSV alignment like 's:s ɛ:ɛ d:d' into triples."""
    if is_missing(raw):
        return []
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            out = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    cost = item[2] if len(item) >= 3 else 0
                    try:
                        cost_f = float(cost)
                    except Exception:
                        cost_f = 0.0
                    out.append((str(item[0]), str(item[1]), cost_f))
            return out
        except Exception:
            pass
    out = []
    for tok in text.split():
        if ":" not in tok:
            continue
        left, right = tok.split(":", 1)
        out.append((left, right, 0.0 if left == right else 1.0))
    return out


def first_last_non_gap(align: list[tuple[str, str, float]], side: int) -> tuple[str, str]:
    vals = []
    for x, y, _c in align:
        seg = x if side == 0 else y
        if seg != "·" and seg:
            vals.append(seg)
    if vals:
        return vals[0], vals[-1]
    return "", ""


def vc_class(seg: str) -> str:
    if not seg:
        return ""
    return "V" if any(ch in VOWELS for ch in seg) else "C"


def intish(x: Any) -> int | None:
    if is_missing(x):
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def derive(row: dict[str, Any]) -> dict[str, Any]:
    e = dict(row)
    base_tier = sget(e, "base_tier", sget(e, "tier", ""))
    e["base_tier"] = base_tier
    align = parse_alignment(e.get("alignment", ""))
    n = len(align)

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

    es = intish(e.get("en_syll"))
    fs = intish(e.get("fr_syll"))
    if es is not None and fs is not None:
        e["en_syll"] = es
        e["fr_syll"] = fs
        e["syllable_delta"] = abs(es - fs)
        e["huge_deletion"] = (es >= 3 and fs <= 1) or (fs >= 3 and es <= 1) or abs(es - fs) >= 3
    else:
        e["syllable_delta"] = 99
        e["huge_deletion"] = True

    score = float(e.get("score", 0.0)) if not is_missing(e.get("score")) else 0.0
    e["score"] = score
    tier = sget(e, "tier", "")
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

    en_on, en_co = first_last_non_gap(align, 0)
    fr_on, fr_co = first_last_non_gap(align, 1)
    e["en_onset"] = sget(e, "en_onset", en_on) or en_on
    e["en_coda"] = sget(e, "en_coda", en_co) or en_co
    e["fr_onset"] = sget(e, "fr_onset", fr_on) or fr_on
    e["fr_coda"] = sget(e, "fr_coda", fr_co) or fr_co
    e["en_onset_class"] = vc_class(e["en_onset"])
    e["en_coda_class"] = vc_class(e["en_coda"])
    e["fr_onset_class"] = vc_class(e["fr_onset"])
    e["fr_coda_class"] = vc_class(e["fr_coda"])
    e["alignment"] = sget(e, "alignment", "")
    e["flags"] = sget(e, "flags", "")
    e["direction"] = sget(e, "direction", "en_fr")
    e["en"] = sget(e, "en", "")
    e["fr"] = sget(e, "fr", "")
    e["en_ipa"] = sget(e, "en_ipa", "")
    e["fr_ipa"] = sget(e, "fr_ipa", "")
    e["pivot"] = sget(e, "pivot", "")
    return e


def build_index(entries: list[dict[str, Any]]) -> dict[str, dict[str, list[int]]]:
    idx = {
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


def write_tsv(entries: list[dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_COLUMNS)
        for x in entries:
            row = []
            for col in TSV_COLUMNS:
                val = x.get(col, "")
                if col in {"huge_deletion", "usable_for_composition"}:
                    val = int(bool(val))
                row.append(val)
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dictionaryv5.tsv")
    ap.add_argument("--tsv", default="dictionary-v5-finalized.tsv")
    ap.add_argument("--index", default="composition-index-finalized.json")
    ap.add_argument("--summary", default="finalize-v5-summary.json")
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    entries = [derive(row) for row in df.to_dict(orient="records")]

    write_tsv(entries, Path(args.tsv))
    Path(args.index).write_text(json.dumps(build_index(entries), ensure_ascii=False), encoding="utf-8")

    base_tiers = Counter(x.get("base_tier", "") for x in entries)
    tiers = Counter(x.get("tier", "") for x in entries)
    usable = sum(1 for x in entries if x.get("usable_for_composition"))
    huge = sum(1 for x in entries if x.get("huge_deletion"))
    summary = {
        "input": args.input,
        "rows": len(entries),
        "base_tiers": dict(base_tiers),
        "tiers": dict(tiers),
        "usable_for_composition": usable,
        "not_usable": len(entries) - usable,
        "huge_deletion": huge,
        "mean_gap_ratio": round(sum(x["gap_ratio"] for x in entries) / len(entries), 4),
        "mean_cheap_gap_ratio": round(sum(x["cheap_gap_ratio"] for x in entries) / len(entries), 4),
        "mean_expensive_gap_ratio": round(sum(x["expensive_gap_ratio"] for x in entries) / len(entries), 4),
        "mean_effective_gap_ratio": round(sum(x["effective_gap_ratio"] for x in entries) / len(entries), 4),
        "outputs": {"tsv": args.tsv, "index": args.index, "summary": args.summary},
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
