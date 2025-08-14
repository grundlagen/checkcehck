from __future__ import annotations
from typing import Callable, List, Tuple, Any, Dict, Optional
from .scoring import CandidateScores

def co_optimize(
    source_text: str,
    initial_literal: str,
    ipa_src: str,
    gen_candidates: Callable[[str], List[Tuple[str, str, Dict[str, Any]]]],
    score_fn: Callable[[str, str, str, str], CandidateScores],
    adjust_literal: Callable[[str, Dict[str, Any]], str],
    max_rounds: int = 2,
    phonetic_floor: float = 0.75,
) -> Dict[str, Any]:
    A = initial_literal
    best: Optional[Dict[str, Any]] = None
    for r in range(max_rounds + 1):
        cands = gen_candidates(A)
        if not cands:
            break
        scored = []
        for B_text, B_ipa, meta in cands:
            s = score_fn(ipa_src, B_text, B_ipa, A)
            scored.append((B_text, s))
        scored.sort(key=lambda x: x[1].score, reverse=True)
        topB, topS = scored[0]
        rec = {"A": A, "B": topB, "scores": topS, "round": r}
        if best is None or topS.score > best["scores"].score:
            best = rec
        if topS.phonetic >= phonetic_floor:
            break
        A = adjust_literal(A, {"toward_src_phones": ipa_src})
    return best or {}