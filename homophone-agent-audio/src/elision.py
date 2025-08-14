import re
from typing import List, Tuple

ELISIONS: List[Tuple[str, str]] = [
    ("le ", "l'"),
    ("la ", "l'"),
    ("de ", "d'"),
    ("ne ", "n'"),
    ("je ", "j'"),
    ("que ", "qu'"),
    ("si il", "s'il"),
]

H_MUET: set[str] = set()

def apply_elisions(s: str) -> str:
    out = s
    vowels = "aeiouâêîôûéèëïüœæAEIOUÂÊÎÔÛÉÈËÏÜŒÆ"
    for a, b in ELISIONS:
        pattern = rf"\b{re.escape(a)}(?=[{vowels}])"
        out = re.sub(pattern, b, out, flags=re.IGNORECASE)
    return out

def allow_liaison(prev_word: str, next_word: str) -> bool:
    return next_word.lower() not in H_MUET