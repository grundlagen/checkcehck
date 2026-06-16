"""
lexicon_index.py — load the FR / EN pronunciation lexica and index them for
homophone hunting.

The lexica are the ``word<TAB>IPA`` TSVs that ship with the homophone agent:
``data/lexique.tsv`` (French, ~246k) and ``data/lexique_en.tsv`` (English, ~65k).

We build, per language:
  * ``by_word``       word -> raw IPA
  * ``by_norm``       normalised-IPA -> set(words)        (exact-sound index)
  * ``buckets``       coarse signature -> list[(word, ipa)]  (near-sound index)
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from . import phon


@dataclass
class Lexicon:
    lang: str
    by_word: Dict[str, str] = field(default_factory=dict)
    by_norm: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    buckets: Dict[str, List[Tuple[str, str]]] = field(default_factory=lambda: defaultdict(list))

    def size(self) -> int:
        return len(self.by_word)


def _looks_alpha(word: str) -> bool:
    # Keep single orthographic words only (skip multiword phrase entries and
    # bare punctuation) so the hunt stays at word granularity.
    if not word or " " in word:
        return False
    return any(c.isalpha() for c in word)


def load_lexicon(path: str, lang: str, *, max_entries: int | None = None,
                 min_phones: int = 2) -> Lexicon:
    lex = Lexicon(lang=lang)
    n = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            word, ipa = parts[0], parts[1]
            if not _looks_alpha(word) or not ipa:
                continue
            phones = phon.seg_ipa(ipa)
            if len(phones) < min_phones:
                continue
            wl = word.lower()
            # keep the first pronunciation seen for a given spelling
            if wl not in lex.by_word:
                lex.by_word[wl] = ipa
            norm = phon.normalise_ipa(ipa)
            lex.by_norm[norm].add(wl)
            lex.buckets[phon.signature(ipa)].append((wl, ipa))
            n += 1
            if max_entries and n >= max_entries:
                break
    return lex


def default_data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "homophone-agent-audio", "data")
