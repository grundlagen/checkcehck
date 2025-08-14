"""
candidate_generation_phrases.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements a phrase‑level candidate generator that uses a
preloaded phrasebank to propose multi‑word homophonic substitutions.
It complements the per‑token BK‑tree substitutions by considering
source n‑grams (up to 5 words) and replacing them with target
phrases of similar pronunciation.  The generator respects a French
lexicon gate: only outputs consisting entirely of known French tokens
(apart from punctuation) are accepted.  Results are deduplicated by
IPA representation.

Functions
~~~~~~~~~

``suggest_phrase_swaps``
    Given a source sentence ``A_text`` and its language codes, return
    up to a specified number of candidate phrase substitutions from the
    phrasebank.  Each candidate is a tuple of (B_text, B_ipa, meta)
    where ``meta`` includes information about the substitution span and
    coverage fraction.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Set

from .phrasebank import PhrasePair
from .orchestrator import g2p

def suggest_phrase_swaps(
    A_text: str,
    lang_src: str,
    lang_tgt: str,
    phrase_pairs: List[PhrasePair],
    en_lex: Set[str] | None,
    fr_lex: Set[str],
    max_spans: int = 50,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Generate candidate substitutions via phrase pairs.

    Args:
        A_text: Source sentence (already literal translation if needed).
        lang_src: Source language code (e.g. "en").  Currently unused.
        lang_tgt: Target language code (e.g. "fr").
        phrase_pairs: List of available phrase pairs matching src→tgt.
        en_lex: Set of known English tokens to use for filtering (can be None).
        fr_lex: Set of known French tokens for gating candidate outputs.
        max_spans: Maximum number of distinct candidates to return.

    Returns:
        A list of (B_text, B_ipa, meta) tuples.  The list is sorted by
        descending span length and coverage fraction.
    """
    toks = A_text.split()
    N = len(toks)
    candidates: List[Tuple[str, str, Dict[str, Any]]] = []
    # Consider n‑grams from longest (5) to shortest (1)
    for n in range(min(5, N), 0, -1):
        for i in range(0, N - n + 1):
            span = " ".join(toks[i : i + n])
            # Find exact matches in the phrasebank
            for pp in phrase_pairs:
                if pp.src == span:
                    # Construct candidate by replacing the span with the tgt phrase
                    B_toks = toks[:i] + [pp.tgt] + toks[i + n :]
                    # Gate: all tokens in B (minus punctuation) must be in FR lexicon
                    def _ok(tok: str) -> bool:
                        tok_clean = tok.strip("’'\"\u2019")
                        return (
                            tok_clean in fr_lex
                            or not tok_clean
                            or set(tok_clean) <= set(".,;:!?-" )
                        )
                    if not all(_ok(t) for t in B_toks):
                        continue
                    B_text = " ".join(B_toks)
                    ipa_B = g2p(B_text, lang_tgt)
                    ipa_src = g2p(A_text, lang_tgt)
                    if not ipa_B:
                        continue
                    span_cov = len(pp.tgt.replace(" ", "")) / max(1, len(ipa_src.replace(" ", "")))
                    candidates.append(
                        (
                            B_text,
                            ipa_B,
                            {
                                "mechanism": "phrasebank",
                                "span": span,
                                "n": n,
                                "span_coverage": span_cov,
                                "pairbank_hit": True,
                            },
                        )
                    )
    # Deduplicate by IPA representation
    uniq: List[Tuple[str, str, Dict[str, Any]]] = []
    seen: Set[str] = set()
    # Sort by longest span, then highest coverage
    for B_text, ipa_B, meta in sorted(
        candidates, key=lambda x: (x[2]["n"], x[2]["span_coverage"]), reverse=True
    ):
        if ipa_B in seen:
            continue
        uniq.append((B_text, ipa_B, meta))
        seen.add(ipa_B)
        if len(uniq) >= max_spans:
            break
    return uniq