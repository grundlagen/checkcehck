from __future__ import annotations
from typing import Iterable, Tuple, Any, List, Dict
# Attempt to import unidecode for ASCII transliteration.  If unavailable,
# define a no‑op fallback that simply returns the input unchanged.  This
# prevents a hard import error in environments without the package.  The
# fallback still allows the rest of the module to function albeit with
# reduced transliteration fidelity.
try:
    from unidecode import unidecode  # type: ignore
except Exception:
    def unidecode(text: str) -> str:  # type: ignore
        """Fallback transliterator: return the input string unchanged."""
        return text

# --- G2P (IPA) ----------------------------------------------------------------
try:
    from phonemizer import phonemize
    _PHONEMIZER_OK = True
except Exception:
    _PHONEMIZER_OK = False

# Lazy‑loaded English IPA dictionary.  When phonemizer is not available we
# attempt to look up each token in this dictionary.  If the dictionary is
# missing or a token is absent, we fall back to a crude ASCII transliteration.
_ENG_IPA_DICT: Dict[str, str] | None = None
_FR_IPA_DICT: Dict[str, str] | None = None

def _load_eng_ipa() -> Dict[str, str]:
    """Load an English IPA dictionary from data/lexique_en.tsv or similar.

    Returns a mapping from lowercase surface forms to IPA strings.  The search
    order includes the module’s data/ directory as well as common parent
    directories.  This function is idempotent and caches its result.
    """
    global _ENG_IPA_DICT
    if _ENG_IPA_DICT is not None:
        return _ENG_IPA_DICT
    ipa_dict: Dict[str, str] = {}
    import os
    # Directories to search for lexique_en.tsv.  Allow override via
    # environment variable DATA_DIR, then try module‑relative paths.
    roots: List[str] = [
        os.environ.get("DATA_DIR", ""),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "data")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data")),
    ]
    candidates: List[str] = []
    for r in roots:
        if not r:
            continue
        for name in ("lexique_en.tsv", "en_UK.txt", "lexique.tsv"):
            candidates.append(os.path.join(r, name))
    # Find the first existing candidate and load it
    for cand in candidates:
        if os.path.exists(cand):
            try:
                with open(cand, "r", encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) < 2:
                            continue
                        word, ipa = parts[0], parts[1]
                        ipa_dict[word.lower()] = ipa
                break
            except Exception:
                continue
    _ENG_IPA_DICT = ipa_dict
    return ipa_dict


def _load_fr_ipa() -> Dict[str, str]:
    """Load a French IPA dictionary from lexique.tsv or similar.

    Returns a mapping from lowercase French surface forms to IPA strings.
    The search order mirrors that used in ``_load_eng_ipa``, but we only
    consider ``lexique.tsv`` and ``fr_FR (1).txt`` if present.  If no
    dictionary is found, an empty map is returned.
    """
    global _FR_IPA_DICT
    if _FR_IPA_DICT is not None:
        return _FR_IPA_DICT
    ipa_dict: Dict[str, str] = {}
    import os
    roots: List[str] = [
        os.environ.get("DATA_DIR", ""),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "data")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data")),
    ]
    candidates: List[str] = []
    for r in roots:
        if not r:
            continue
        # The primary French lexicon is lexique.tsv.  If a backup exists under a
        # different name (e.g., fr_FR (1).txt), include it in the search.  We
        # also look at lexique_en.tsv as a fallback in case a single file is
        # provided with mixed languages.
        for name in ("lexique.tsv", "fr_FR (1).txt", "lexique_en.tsv"):
            candidates.append(os.path.join(r, name))
    for cand in candidates:
        if os.path.exists(cand):
            try:
                with open(cand, "r", encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) < 2:
                            continue
                        word, ipa = parts[0], parts[1]
                        ipa_dict[word.lower()] = ipa
                break
            except Exception:
                continue
    _FR_IPA_DICT = ipa_dict
    return ipa_dict


def g2p(text: str, lang: str) -> str:
    """Return normalized IPA for text.

    If `phonemizer` is available, it is used for all languages.  Otherwise,
    when translating English (`lang == 'en'`), we look up each token in a
    locally provided IPA dictionary.  For unseen tokens or other languages,
    we fall back to a crude ASCII transliteration via `unidecode`.  The
    returned string is normalized to have single spaces between tokens.
    """
    if not text.strip():
        return ""
    # Use phonemizer when available
    if _PHONEMIZER_OK:
        ipa = phonemize(
            text,
            language=lang,
            backend="espeak",
            strip=True,
            njobs=1,
            punctuation_marks=';:,.!?¡¿—…"«»“”()',
            with_stress=False,
            preserve_punctuation=False,
        )
        return " ".join(ipa.split())
    # Fallback behaviour when phonemizer is not available.  Attempt to
    # transliterate tokens using a language‑specific IPA dictionary; if a
    # token is not found, fall back to naive ASCII transliteration via
    # ``unidecode``.  Unknown languages fall back to ASCII.
    lang_lc = lang.lower()
    if lang_lc.startswith("en"):
        eng_dict = _load_eng_ipa()
        ipa_parts: List[str] = []
        for tok in text.split():
            ipa_tok = eng_dict.get(tok.lower())
            if ipa_tok:
                ipa_parts.append(ipa_tok)
            else:
                ipa_parts.append(unidecode(tok.lower()))
        return " ".join(ipa_parts)
    elif lang_lc.startswith("fr"):
        fr_dict = _load_fr_ipa()
        ipa_parts: List[str] = []
        for tok in text.split():
            ipa_tok = fr_dict.get(tok.lower())
            if ipa_tok:
                ipa_parts.append(ipa_tok)
            else:
                ipa_parts.append(unidecode(tok.lower()))
        return " ".join(ipa_parts)
    else:
        # Generic fallback: ASCII transliteration
        ipa = unidecode(text.lower())
        return " ".join(ipa.split())

# --- Translation (stub – keep literal unless you wire an API) -----------------
def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    # TODO: swap with a real MT call; for now echo source as “literal”
    return text

# --- Candidate generation hook ------------------------------------------------
from .candidate_generation import suggest_near_pronunciations
def generate_candidates(A_text: str, ipa_src: str, bk_tree, per_token_k: int = 3):
    return suggest_near_pronunciations(A_text, ipa_src, bk_tree, per_token_k=per_token_k)

# --- Semantic / fluency (stubs; keep fast) -----------------------------------
def semantic_score(a_text: str, b_text: str) -> float:
    # lightweight overlap proxy; replace with a proper model later
    a, b = set(a_text.lower().split()), set(b_text.lower().split())
    return 0.0 if not a else len(a & b) / max(1, len(a))

def fluency_score(text: str) -> float:
    # quick heuristic: penalize too many short tokens/punctuation
    toks = text.split()
    if not toks: return 0.0
    avg = sum(len(t) for t in toks) / len(toks)
    return max(0.0, min(1.0, (avg - 2.0) / 4.0))  # 0..1

# --- BK-tree loading ----------------------------------------------------------
from .lexicon import PhoneBKTree, load_lexique
# Import feature‑based similarity for BK‑tree bucketization
try:
    from .phone_metric import similarity as _phone_sim
except Exception:
    _phone_sim = None  # type: ignore

def load_bk_tree(entries: Iterable[Tuple[str, str]]) -> PhoneBKTree:
    """Build a BK‑tree using a feature‑weighted metric when available."""
    # Use our phone similarity to derive buckets if available; otherwise None
    metric = _phone_sim if _phone_sim is not None else None
    bk = PhoneBKTree(metric=metric)
    for w, ipa in entries:
        if ipa:
            bk.add(ipa, w)
    return bk