"""
journal.py — the memory that lets the map *whittle* across routines.

Every resonance the hunt evaluates is written as one JSONL record.  Successive
routines load what has already been judged, so the frontier of unexplained
sound-coincidences shrinks while the reservoir of confirmed sound+meaning
matches grows.  This mirrors the existing ``evolutionary_journal.jsonl`` habit
in this repo: append-only, line-delimited, greppable.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Iterable, Tuple

PairKey = Tuple[str, str]  # (en_word, fr_word)


def pair_key(en_word: str, fr_word: str) -> str:
    return f"{en_word.lower()}↔{fr_word.lower()}"


def load_seen(path: str) -> Dict[str, dict]:
    """Return the latest record per pair key (last write wins)."""
    seen: Dict[str, dict] = {}
    if not os.path.exists(path):
        return seen
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "key" in rec:
                seen[rec["key"]] = rec
    return seen


def append(path: str, records: Iterable[dict]) -> int:
    n = 0
    with open(path, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
