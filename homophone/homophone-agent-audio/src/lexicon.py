from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Iterable

try:
    from rapidfuzz.distance import Levenshtein
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

def _levenshtein(s1: str, s2: str) -> int:
    if _RAPIDFUZZ_AVAILABLE:
        return Levenshtein.distance(s1, s2)
    # Fallback DP
    len1, len2 = len(s1), len(s2)
    dp = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len2 + 1):
            temp = dp[j]
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[len2]

class PhoneBKTree:
    """A BK-tree over IPA strings.

    When a similarity function is provided, integer distance buckets are
    derived by rounding ``(1 − similarity) * 10``.  Without a metric,
    plain Levenshtein distance is used.  Each node stores the IPA and
    its associated word.  Edges store child indices and their distance
    to the parent.
    """

    def __init__(self, metric: Optional[callable] = None) -> None:
        # Each node is a tuple (ipa, word)
        self.nodes: List[Tuple[str, str]] = []
        # Each entry in edges maps node index to list of (child_index, dist)
        self.edges: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        # Similarity metric returning a float in [0,1]; if None, Levenshtein is used
        self._sim = metric

    def add(self, ipa: str, word: str) -> None:
        if not self.nodes:
            self.nodes.append((ipa, word))
            return
        idx = 0
        while True:
            current_ipa, _ = self.nodes[idx]
            # Compute distance bucket
            if self._sim is None:
                d = _levenshtein(ipa, current_ipa)
            else:
                # Convert similarity to integer bucket 0..10
                sim = self._sim(ipa, current_ipa)
                # Round to nearest int; ensure non‑negative
                d = int(round((1.0 - sim) * 10))
            next_idx = None
            for child_idx, child_d in self.edges[idx]:
                if child_d == d:
                    next_idx = child_idx
                    break
            if next_idx is not None:
                idx = next_idx
            else:
                new_idx = len(self.nodes)
                self.nodes.append((ipa, word))
                self.edges[idx].append((new_idx, d))
                break

    def query(self, ipa: str, max_d: int = 3, topk: int = 10) -> List[Tuple[int, str, str]]:
        if not self.nodes:
            return []
        candidates = [(0, 0)]
        results: List[Tuple[int, str, str]] = []
        while candidates:
            node_idx, _ = candidates.pop()
            node_ipa, node_word = self.nodes[node_idx]
            # Compute distance bucket
            if self._sim is None:
                dist = _levenshtein(ipa, node_ipa)
            else:
                sim = self._sim(ipa, node_ipa)
                dist = int(round((1.0 - sim) * 10))
            if dist <= max_d:
                results.append((dist, node_word, node_ipa))
            for child_idx, child_d in self.edges[node_idx]:
                if dist - max_d <= child_d <= dist + max_d:
                    candidates.append((child_idx, child_d))
        results.sort(key=lambda x: x[0])
        return results[:topk]

def load_lexique(path: str) -> Iterable[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            if len(parts) >= 2:
                word = parts[0]
                ipa = parts[1]
                yield word, ipa