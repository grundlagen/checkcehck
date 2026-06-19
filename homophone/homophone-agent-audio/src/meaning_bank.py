"""
meaning_bank.py
~~~~~~~~~~~~~~~

The *second pair bank* of the project.

Where ``phrasebank.py`` / ``pairbank.tsv`` give us the **homophone bank**
(EN surface forms that *sound like* FR surface forms), this module gives us
the **meaning bank**: a FR -> EN gloss table and an EN -> EN synonym table.

Together these two banks are the substrate for *semantic chaining*: an English
word can reach a French word by **sound** (homophone bank), and that French
word can reach an English meaning by **gloss** (meaning bank).  When the sound
path and the meaning path land on the *same* English concept (directly, or via
synonyms), we have found a **resonant bridge** — a cross-lingual pun that also
tells the truth.  See ``semantic_chain.py`` for the engine that walks these
chains.

Design notes
------------
* Everything is pure-Python and degrades gracefully, matching the rest of the
  codebase.  The banks load from TSV files in ``data/`` and can be overridden
  via the ``DATA_DIR`` environment variable.
* The seed TSVs (``meaningbank.tsv``, ``synonyms_en.tsv``) are intentionally
  small but real.  They exist to *bootstrap* the hunt and to make the demo
  produce honest, inspectable output.  The grand goal is to grow these banks
  (LLM-assisted, Wiktionary, WordNet, bilingual dictionaries) until the map of
  meaning approaches completeness.  The loaders are written so that swapping in
  a 100k-row file requires no code changes.
* An optional WordNet hook (``nltk``) is consulted for synonyms when available;
  it is purely additive on top of the seed table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import os
import re
import unicodedata


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def normalize(s: str) -> str:
    """Lowercase, strip diacritics-as-noise-tolerant, collapse whitespace.

    We keep base letters and apostrophes (French elision: ``c'est``) but drop
    surrounding punctuation.  Diacritics are *preserved* for French surface
    forms when present, because ``mère`` and ``mer`` are different words; the
    homophone bank already encodes the sound relationship, so the meaning bank
    keys on the exact written form.
    """
    s = unicodedata.normalize("NFC", s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \t\n\r.,;:!?\"«»“”()")


def _data_roots() -> List[str]:
    here = os.path.dirname(__file__)
    return [
        os.environ.get("DATA_DIR", ""),
        os.path.abspath(os.path.join(here, "..", "data")),
        os.path.abspath(os.path.join(here, "data")),
        os.path.abspath(os.path.join(here, "..", "..", "data")),
    ]


def _find(filename: str) -> Optional[str]:
    for root in _data_roots():
        if not root:
            continue
        cand = os.path.join(root, filename)
        if os.path.exists(cand):
            return cand
    return None


# --------------------------------------------------------------------------- #
# Meaning bank: FR surface -> set of EN glosses
# --------------------------------------------------------------------------- #
@dataclass
class MeaningBank:
    """FR -> EN gloss lookup.

    A French surface form may carry several senses; we store all of them as a
    flat set of English gloss tokens/phrases.  ``gloss`` returns the set (or an
    empty set when the word is unknown — an honest "I don't know yet").
    """

    fr_to_en: Dict[str, Set[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, filename: str = "meaningbank.tsv") -> "MeaningBank":
        path = _find(filename)
        fr_to_en: Dict[str, Set[str]] = {}
        if path:
            with open(path, "r", encoding="utf-8") as fh:
                header = fh.readline()  # skip header
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    fr = normalize(parts[0])
                    glosses = {normalize(g) for g in parts[1].split(",") if g.strip()}
                    if not fr or not glosses:
                        continue
                    fr_to_en.setdefault(fr, set()).update(glosses)
        return cls(fr_to_en=fr_to_en)

    def gloss(self, fr_word: str) -> Set[str]:
        return set(self.fr_to_en.get(normalize(fr_word), set()))

    def __len__(self) -> int:
        return len(self.fr_to_en)


# --------------------------------------------------------------------------- #
# Synonym bank: EN word -> set of EN synonyms (symmetric closure)
# --------------------------------------------------------------------------- #
@dataclass
class SynonymBank:
    """EN -> EN synonym lookup with optional WordNet augmentation."""

    syn: Dict[str, Set[str]] = field(default_factory=dict)
    _wordnet_ok: bool = False

    @classmethod
    def load(cls, filename: str = "synonyms_en.tsv", use_wordnet: bool = True) -> "SynonymBank":
        path = _find(filename)
        syn: Dict[str, Set[str]] = {}

        def link(a: str, b: str) -> None:
            a, b = normalize(a), normalize(b)
            if not a or not b or a == b:
                return
            syn.setdefault(a, set()).add(b)
            syn.setdefault(b, set()).add(a)  # symmetric closure

        if path:
            with open(path, "r", encoding="utf-8") as fh:
                fh.readline()  # header
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    head = parts[0]
                    for s in parts[1].split(","):
                        link(head, s)

        wordnet_ok = False
        if use_wordnet:
            try:  # purely additive; never required
                from nltk.corpus import wordnet as wn  # type: ignore

                _ = wn.synsets("test")  # trigger lazy data load / fail fast
                wordnet_ok = True
            except Exception:
                wordnet_ok = False

        return cls(syn=syn, _wordnet_ok=wordnet_ok)

    def synonyms(self, word: str) -> Set[str]:
        word = normalize(word)
        out: Set[str] = set(self.syn.get(word, set()))
        if self._wordnet_ok:
            try:
                from nltk.corpus import wordnet as wn  # type: ignore

                for syns in wn.synsets(word):
                    for lemma in syns.lemmas():
                        name = lemma.name().replace("_", " ").lower()
                        if name != word:
                            out.add(name)
            except Exception:
                pass
        return out

    def expand(self, word: str) -> Set[str]:
        """The word together with its synonyms — the meaning neighbourhood."""
        return {normalize(word)} | self.synonyms(word)
