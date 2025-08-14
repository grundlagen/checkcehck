"""
phrasebank.py
~~~~~~~~~~~~~

This module defines a simple typed container and loader for
multi‑word homophonic phrase pairs.  The TSV file format expected
matches the ``pairbank.tsv`` generated offline from your uploaded
CSV files.  Each row contains columns:

    src, src_ipa, src_lang, tgt, tgt_ipa, tgt_lang, tag

Only pairs matching the desired direction (e.g. ("en","fr")) are
loaded.  Multi‑word phrases are preserved as a single string with
spaces.  IPA values may be empty if unknown.

The loader normalizes Unicode by removing combining characters and
collapsing whitespace.  You can extend the ``_norm`` function if
additional normalization is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Iterable
import unicodedata
import re

@dataclass(frozen=True)
class PhrasePair:
    """A pair of homophonic phrases with optional IPA and tag."""
    src: str
    src_ipa: str
    src_lang: str
    tgt: str
    tgt_ipa: str
    tgt_lang: str
    tag: str

def _norm(s: str) -> str:
    """Normalize a string by stripping zero‑width and collapsing whitespace."""
    # Remove combining marks (including zero‑width joiners) and normalize NFC
    s = "".join(
        ch for ch in unicodedata.normalize("NFKD", s) if unicodedata.category(ch) != 'Cf'
    )
    # Collapse multiple whitespace to single spaces and trim
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_phrasebank(path: str, direction: Tuple[str, str] = ("en", "fr")) -> List[PhrasePair]:
    """Load phrase pairs from a TSV, filtering by language direction.

    This loader is tolerant to missing optional columns (e.g., src_ipa, tgt_ipa)
    and will substitute empty strings when they are absent.  Each row of the
    TSV may have more columns than required (e.g., source_file); these are
    ignored.  Only pairs matching ``direction`` (src_lang, tgt_lang) are
    returned.
    """
    src_dir, tgt_dir = direction
    out: List[PhrasePair] = []
    with open(path, "r", encoding="utf-8") as fh:
        # Parse header and build index map; missing columns are noted
        header = next(fh).rstrip("\n").split("\t")
        idx: Dict[str, int] = {h: i for i, h in enumerate(header)}
        # Helper to fetch a value or return empty string if column missing or index out of range
        def _col(cols: List[str], name: str) -> str:
            pos = idx.get(name)
            if pos is None or pos >= len(cols):
                return ""
            return cols[pos]
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            # Skip if language direction doesn't match
            src_lang = _col(cols, "src_lang")
            tgt_lang = _col(cols, "tgt_lang")
            if (src_lang, tgt_lang) != (src_dir, tgt_dir):
                continue
            src = _norm(_col(cols, "src"))
            tgt = _norm(_col(cols, "tgt"))
            src_ipa = _norm(_col(cols, "src_ipa"))
            tgt_ipa = _norm(_col(cols, "tgt_ipa"))
            tag = _col(cols, "tag") or ""
            # Append phrase pair, substituting empty strings for missing IPAs
            out.append(
                PhrasePair(
                    src,
                    src_ipa,
                    src_lang,
                    tgt,
                    tgt_ipa,
                    tgt_lang,
                    tag,
                )
            )
    return out