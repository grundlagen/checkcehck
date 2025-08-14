"""Simple complexity metric (CORT).

The :func:`cort_score` function provides a lightweight heuristic for
measuring the linguistic or structural complexity of a candidate text.
Here we use the variance of token lengths as a proxy for complexity:
strings with more variation in word length are treated as more complex.
The variance is mapped into ``[0, 1]`` via ``var / (var + 1)``.

The metric is intentionally simple and does not require external
libraries so that it can run in constrained environments.
"""
from __future__ import annotations

from typing import List


def cort_score(candidate_text: str) -> float:
    """Return a complexity score for ``candidate_text``.

    The score is based on the variance of token lengths. Higher variance
    implies higher complexity. The raw variance is normalised to the
    range ``[0, 1]`` using ``var / (var + 1)``.

    Args:
        candidate_text: Text for which to compute the complexity score.

    Returns:
        A float between ``0`` (minimal complexity) and ``1`` (high
        complexity).
    """
    tokens: List[str] = [t for t in candidate_text.split() if t]
    if not tokens:
        return 0.0
    lengths = [len(t) for t in tokens]
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    return max(0.0, min(1.0, variance / (variance + 1)))
