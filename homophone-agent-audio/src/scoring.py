from __future__ import annotations
from dataclasses import dataclass

@dataclass
class CandidateScores:
    phonetic: float
    semantic: float
    fluency: float
    prosody: float
    score: float

def combine_scores(
    phonetic: float,
    semantic: float,
    fluency: float,
    prosody: float = 0.0,
    w_ph: float = 0.45,
    w_sem: float = 0.30,
    w_flu: float = 0.15,
    w_pros: float = 0.10,
) -> CandidateScores:
    combined = w_ph * phonetic + w_sem * semantic + w_flu * fluency + w_pros * prosody
    return CandidateScores(
        phonetic=phonetic,
        semantic=semantic,
        fluency=fluency,
        prosody=prosody,
        score=combined,
    )