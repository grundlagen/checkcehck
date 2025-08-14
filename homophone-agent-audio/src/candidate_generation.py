from __future__ import annotations
from typing import List, Tuple, Dict, Any
from rapidfuzz.distance import Levenshtein

from .orchestrator import g2p
from .elision import apply_elisions
from .phonetics import apply_tricks
from .lexicon import PhoneBKTree

def _compose_tokens(tokens: List[str], idx: int, repl_word: str) -> str:
    out = tokens[:]
    out[idx] = repl_word
    text = " ".join(out)
    return apply_elisions(text)

def suggest_near_pronunciations(A_text: str, ipa_src: str, bk_tree: PhoneBKTree, per_token_k: int = 3) -> List[Tuple[str, str, Dict[str, Any]]]:
    tokens = A_text.split()
    ipas = [g2p(t, "fr") for t in tokens]

    per_position: List[List[Tuple[str,str,Dict[str,Any]]]] = []
    for tkn, ipa in zip(tokens, ipas):
        neigh = bk_tree.query(ipa, max_d=2, topk=per_token_k) if ipa else []
        cand_words = [(word, n_ipa, {"src": tkn, "ipa_src": ipa, "ipa_neigh": n_ipa, "d": d})
                      for (d, word, n_ipa) in neigh]
        if not cand_words:
            cand_words = [(tkn, ipa, {"src": tkn, "ipa_src": ipa, "ipa_neigh": ipa, "d": 0})]
        per_position.append(cand_words)

    candidates: List[Tuple[str,str,Dict[str,Any]]] = []
    for idx in range(len(tokens)):
        for (w, w_ipa, meta) in per_position[idx]:
            cand_text = _compose_tokens(tokens, idx, w)
            cand_ipa = g2p(cand_text, "fr")
            for t_text, t_ipa, note in apply_tricks(cand_text, cand_ipa):
                m = {**meta, "pos": idx, "trick": note}
                candidates.append((t_text, t_ipa, m))

    uniq: List[Tuple[str,str,Dict[str,Any]]] = []
    seen = []
    for txt, ipa, m in candidates:
        if not ipa:
            continue
        ok = True
        for s in seen:
            if Levenshtein.distance(ipa, s) <= 1:
                ok = False
                break
        if ok:
            uniq.append((txt, ipa, m))
            seen.append(ipa)
    return uniq[:50]