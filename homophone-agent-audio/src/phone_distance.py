from __future__ import annotations

import re
try:
    import panphon.distance as ppd
    _PANPHON_OK = True
except Exception:
    _PANPHON_OK = False

# Fallback feature‑based metric from phone_metric
try:
    from .phone_metric import feature_distance as _feat_dist
    from .phone_metric import similarity as _feat_sim
    _FEAT_METRIC_OK = True
except Exception:
    _FEAT_METRIC_OK = False

from difflib import SequenceMatcher

VOWELS = "aeiouyɑɛœøəɔæɥɪʏʊ̃̃ɑ̃ɛ̃ɔ̃œ̃"

def _sylls(ipa: str):
    if "." in ipa:
        return [s for s in ipa.split(".") if s]
    return re.findall(r"[^\s]+", ipa) or [ipa]

def _onc(ipa_syll: str):
    m = re.match(rf"^([^{VOWELS}]*)([{VOWELS}]+)(.*)$", ipa_syll)
    if not m:
        return "", ipa_syll, ""
    return m.group(1), m.group(2), m.group(3)

class PhoneDistance:
    def __init__(self, onset_w: float = 0.1, nucleus_w: float = 0.2, coda_w: float = 0.1):
        self.onset_w, self.nucleus_w, self.coda_w = onset_w, nucleus_w, coda_w
        self.dist = ppd.Distance() if _PANPHON_OK else None

    def _base(self, a: str, b: str) -> float:
        """Return a normalized base distance between two IPA strings."""
        if self.dist:
            # Use panphon's feature edit distance if available
            try:
                return self.dist.feature_edit_distance(a, b, norm=True)
            except Exception:
                pass
        if _FEAT_METRIC_OK:
            # Use feature‑weighted fallback
            try:
                return _feat_dist(a, b)
            except Exception:
                pass
        # Fallback to SequenceMatcher ratio (very coarse)
        return 1.0 - SequenceMatcher(None, a, b).ratio()

    def _bonus(self, a: str, b: str) -> float:
        sa, sb = _sylls(a), _sylls(b)
        n = min(len(sa), len(sb))
        bon = 0.0
        for i in range(n):
            oa, na, ca = _onc(sa[i])
            ob, nb, cb = _onc(sb[i])
            if oa and ob and oa[:1] == ob[:1]: bon += self.onset_w
            if na and nb and na[:1] == nb[:1]: bon += self.nucleus_w
            if ca and cb and ca[-1:] == cb[-1:]: bon += self.coda_w
        return min(bon, 0.6)

    def similarity(self, a_ipa: str, b_ipa: str) -> float:
        """Return a phonetic similarity score in [0, 1]."""
        sim = 1.0 - self._base(a_ipa, b_ipa) + self._bonus(a_ipa, b_ipa)
        return max(0.0, min(1.0, sim))

# A small dataclass to return scores together with the combined score
from dataclasses import dataclass
@dataclass
class ScoreBreakdown:
    phonetic: float
    semantic: float
    fluency: float
    prosody: float
    cort: float
    score: float