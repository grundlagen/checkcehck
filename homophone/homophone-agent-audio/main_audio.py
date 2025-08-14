#!/usr/bin/env python3
"""
Command line interface for the homophonic–literal dual translation agent
with audio‑assisted scoring and embedding‑based semantics.

This script extends ``main.py`` by incorporating two enhancements:

1. **Embedding‑based semantic similarity:** Instead of the simple
   token‑overlap heuristic used in the base model, semantic similarity
   is computed via sentence embeddings when the optional
   ``sentence_transformers`` package is available. When unavailable,
   the heuristic is used as a fallback. See ``src/embedding.py`` for
   details.

2. **Audio‑assisted "heard‑as" bonus:** Optionally, for each candidate
   homophonic translation the script synthesizes the candidate text
   into audio, runs speech recognition on the result, and compares the
   ASR transcript back to the source text and IPA. If the ASR
   transcript is sufficiently close, a small bonus is added to the
   overall score, favouring candidates that not only look similar on
   paper but also sound convincingly like the source when spoken.
   Because this environment lacks real TTS/ASR backends, the default
   implementation of this bonus always returns zero. Users should
   override the functions in ``src/audio_helpers.py`` to integrate
   their own services.

Usage example::

    python main_audio.py \
        --src-text "the night rate" \
        --src-lang en \
        --tgt-lang fr \
        --use-audio-check \
        --bonus-value 0.08

The optional ``--use-audio-check`` flag enables the audio loop. The
``--bonus-value`` flag controls how much to add to the score when a
candidate passes the heard‑as test. The default bonus is 0.05.

Note that meaningful results require a populated lexicon file
(``lexique.tsv``) in the ``data/`` directory and replacement of the
stub translation and phonetic conversion functions in
``src/orchestrator.py`` with real implementations.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

from src.orchestrator import (
    translate,
    g2p,
    generate_candidates,
    fluency_score,
    load_bk_tree,
)
from src.phrasebank import load_phrasebank, PhrasePair  # type: ignore
from src.candidate_generation_phrases import suggest_phrase_swaps  # type: ignore
from src.lexicon import load_lexique
from src.phone_distance import PhoneDistance, ScoreBreakdown
from src.scoring import combine_scores
from src.judge import judge, judge_tts
from src.audio_helpers import heard_as_bonus


def load_default_lexicon(data_dir: str) -> Iterable[Tuple[str, str]]:
    """Load lexicon entries from ``data_dir`` if present.

    This function mirrors the helper in ``main.py``. It expects a
    ``lexique.tsv`` file in the given directory, containing tab
    separated columns ``word`` and ``ipa``. Lines without IPA are
    skipped. If no such file exists, an empty list is returned.
    """
    lex_path = os.path.join(data_dir, "lexique.tsv")
    if os.path.isfile(lex_path):
        print(f"Loading lexicon from {lex_path}...")
        return load_lexique(lex_path)
    else:
        print(f"No lexicon found at {lex_path}; BK‑tree will be empty.")
        return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Homophonic–Literal Dual Translation Agent with "
            "embedding semantics and optional audio‑assisted scoring"
        )
    )
    parser.add_argument("--src-text", required=True, help="Source text to translate")
    parser.add_argument(
        "--src-lang", default="en", help="Source language code (default: en)"
    )
    parser.add_argument(
        "--tgt-lang", default="fr", help="Target language code (default: fr)"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Directory containing lexicon files (default: ./data)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="Number of co‑optimization rounds (0 disables co‑optimization)",
    )
    parser.add_argument(
        "--phonetic-threshold",
        type=float,
        default=0.75,
        help="Desired phonetic score threshold for co‑optimization",
    )
    parser.add_argument(
        "--use-audio-check",
        action="store_true",
        help="Enable the audio‑assisted heard‑as bonus",
    )
    parser.add_argument(
        "--bonus-value",
        type=float,
        default=0.05,
        help="Score bonus to apply when a candidate passes the heard‑as test",
    )
    parser.add_argument(
        "--cort-weight",
        type=float,
        default=0.0,
        help="Weight for CORT complexity score in overall combination",
    )
    parser.add_argument(
        "--pairbank",
        default=None,
        help=(
            "Path to a TSV file containing phrase pairs (columns src, src_ipa, src_lang, tgt, tgt_ipa, tgt_lang, tag)."
        ),
    )
    parser.add_argument(
        "--no-cognates",
        action="store_true",
        help="If set, remove candidates that contain tokens very similar to the source (cognates)",
    )
    args = parser.parse_args()

    src_text = args.src_text
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang

    # Load BK‑tree from data
    entries = load_default_lexicon(args.data_dir)
    bk_tree = load_bk_tree(entries)
    # Build a French lexicon set for gating phrase candidates
    fr_lex_set = {w for (w, ipa) in entries if w}
    # Optionally load phrasebank if provided
    phrase_pairs: List[PhrasePair] = []
    if args.pairbank:
        try:
            phrase_pairs = load_phrasebank(args.pairbank, (args.src_lang, args.tgt_lang))
        except Exception as e:
            print(f"Warning: could not load phrasebank {args.pairbank}: {e}")

    # Perform literal translation
    A_text = translate(src_text, src_lang, tgt_lang)
    # Compute IPA for source and literal translation
    ipa_src = g2p(src_text, src_lang)
    ipa_A = g2p(A_text, tgt_lang)

    # Instantiate phonetic distance
    phone_dist = PhoneDistance()

    # Scoring function wrapper
    def score_candidate(B_text: str, B_ipa: str) -> ScoreBreakdown:
        # Compute component scores and combine using existing weighting
        comp = judge(src_text, ipa_src, A_text, B_text, B_ipa, phone_dist)
        base_raw = combine_scores(
            comp["phonetic"],
            comp["semantic"],
            comp["fluency"],
            0.0,
            comp.get("cort", 0.0),
            w_cort=args.cort_weight,
        )
        # Optionally apply audio bonus; reconfirmation is not a hard gate because
        # the default audio helpers are stubs.  Clients can override
        # ``heard_as_bonus`` to integrate real TTS/ASR services.  We still call
        # ``judge_tts`` so that callers can inspect the reconfirmation
        # likelihood if desired, but we do not drop candidates solely on its
        # output.
        if args.use_audio_check:
            _reconf_score = judge_tts(
                src_text,
                ipa_src,
                B_text,
                B_ipa,
                phone_dist,
                bonus_threshold=args.bonus_value,
            )
            bonus = heard_as_bonus(
                src_text,
                ipa_src,
                B_text,
                B_ipa,
                phone_dist,
                bonus=args.bonus_value,
            )
            if bonus > 0.0:
                return ScoreBreakdown(
                    base_raw.phonetic,
                    base_raw.semantic,
                    base_raw.fluency,
                    base_raw.prosody,
                    base_raw.cort,
                    min(1.0, base_raw.score + bonus),
                )
        return ScoreBreakdown(
            base_raw.phonetic,
            base_raw.semantic,
            base_raw.fluency,
            base_raw.prosody,
            base_raw.cort,
            base_raw.score,
        )

    # Candidate generation function
    def gen_cands(A: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Generate candidate translations for literal ``A``.

        Combines per‑token BK‑tree substitutions with multi‑word phrase
        substitutions from the phrasebank when available.  Duplicate IPA
        outputs are pruned.
        """
        cands: List[Tuple[str, str, Dict[str, Any]]] = []
        # Per‑token homophones
        cands += generate_candidates(A, ipa_src, bk_tree, per_token_k=7)
        # Phrase substitutions
        if phrase_pairs:
            # Attempt to load an English lexicon set if available for filtering; may be empty
            en_lex_set: set[str] = set()
            try:
                # Try to read English lexicon from data/lexique_en.tsv if present
                en_path = os.path.join(args.data_dir, "lexique_en.tsv")
                if os.path.isfile(en_path):
                    with open(en_path, "r", encoding="utf-8") as fh:
                        for line in fh:
                            parts = line.strip().split("\t")
                            if parts:
                                en_lex_set.add(parts[0])
            except Exception:
                en_lex_set = set()
            cands += suggest_phrase_swaps(
                A,
                args.src_lang,
                args.tgt_lang,
                phrase_pairs,
                en_lex_set,
                fr_lex_set,
                max_spans=50,
            )
        # Deduplicate by IPA (B_ipa)
        uniq: List[Tuple[str, str, Dict[str, Any]]] = []
        seen_ipa: set[str] = set()
        for B_text, B_ipa, meta in cands:
            if B_ipa in seen_ipa:
                continue
            # Optionally remove cognates at generation stage
            if args.no_cognates:
                # Compare each token with src_text tokens; skip if too similar
                skip = False
                src_tokens = src_text.split()
                tgt_tokens = B_text.split()
                from src.cognate import is_cognateish  # local import to avoid circular
                for a, b in zip(src_tokens, tgt_tokens):
                    if is_cognateish(a, b, thresh=0.85):
                        skip = True
                        break
                if skip:
                    continue
            uniq.append((B_text, B_ipa, meta))
            seen_ipa.add(B_ipa)
        return uniq[:100]

    # Adjustment function for co‑optimization (stub: identity)
    def adjust_A(A: str, guidance: Dict[str, Any]) -> str:
        # Real implementation might tweak synonyms to improve phonetic alignment
        return A

    # Generate B candidates
    candidates = gen_cands(A_text)
    if not candidates:
        # Fallback: use literal translation as the only candidate
        candidates = [(A_text, ipa_A, {})]

    # Co‑optimization loop: optional, akin to main.py
    best_B = None  # type: ignore
    best_B_ipa = None  # type: ignore
    best_scores = None  # type: ignore
    A_current = A_text
    candidates_current = candidates
    for step in range(max(1, args.max_rounds or 1)):
        scored: List[Tuple[str, str, ScoreBreakdown]] = []
        for B_text, B_ipa, _meta in candidates_current:
            s = score_candidate(B_text, B_ipa)
            scored.append((B_text, B_ipa, s))
        scored.sort(key=lambda x: x[2].score, reverse=True)
        best_B, best_B_ipa, best_scores = scored[0]
        # Early break if phonetic score meets threshold
        if best_scores.phonetic >= args.phonetic_threshold:
            break
        # Otherwise adjust A and regenerate candidates
        A_current = adjust_A(A_current, {"best": best_B, "phon": best_scores.phonetic})
        ipa_A_current = g2p(A_current, tgt_lang)
        candidates_current = gen_cands(A_current)
        if not candidates_current:
            candidates_current = [(A_current, ipa_A_current, {})]

    # Compile alternates list (excluding best)
    alt_list: List[Dict[str, Any]] = []
    for B_text, B_ipa, s in sorted(
        [
            (B_text, B_ipa, score_candidate(B_text, B_ipa))
            for (B_text, B_ipa, _meta) in candidates
        ],
        key=lambda x: x[2].score,
        reverse=True,
    ):
        if best_B is not None and B_text == best_B:
            continue
        alt_list.append(
            {
                "text": B_text,
                "ipa": B_ipa,
                "scores": {
                    "phonetic": round(s.phonetic, 3),
                    "semantic": round(s.semantic, 3),
                    "fluency": round(s.fluency, 3),
                    "prosody": round(s.prosody, 3),
                    "cort": round(s.cort, 3),
                    "score": round(s.score, 3),
                },
            }
        )

    # Build result object
    result = {
        "literal": A_text,
        "homophonic": best_B,
        "ipa_src": ipa_src,
        "ipa_literal": ipa_A,
        "ipa_homophonic": best_B_ipa,
        "scores": {
            "phonetic": round(best_scores.phonetic, 3) if best_scores else None,
            "semantic": round(best_scores.semantic, 3) if best_scores else None,
            "fluency": round(best_scores.fluency, 3) if best_scores else None,
            "prosody": round(best_scores.prosody, 3) if best_scores else None,
            "cort": round(best_scores.cort, 3) if best_scores else None,
            "score": round(best_scores.score, 3) if best_scores else None,
        },
        "alternates": alt_list,
        "notes": (
            "Embedding semantics are used when available; audio bonus is "
            + ("enabled" if args.use_audio_check else "disabled")
        ),
    }

    # Print JSON result
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
