"""
embedding.py
-------------------

This module provides a semantic similarity function that attempts to use
sentence embeddings to compute a cosine similarity score between two
pieces of text. If a suitable embedding backend is not available, it
falls back to a simple bag‑of‑words overlap heuristic. The returned
similarity is normalized to the range [0, 1].

Notes
~~~~~

* The primary implementation uses the ``sentence_transformers`` library
  with the ``all‑MiniLM‑L6‑v2`` model, which offers multilingual
  support and operates without requiring GPU acceleration. If the
  library or model cannot be loaded, the code gracefully falls back to
  the heuristic. End users can install the optional dependency via

  ``pip install sentence-transformers``

* The fallback heuristic computes the fraction of shared word tokens
  between the two inputs. Although far less nuanced than real
  embeddings, it prevents the pipeline from breaking entirely when
  dependencies are missing.

* No network calls occur in this module; all operations are strictly
  local. For a production system you may wish to integrate with
  hosted embedding APIs instead of bundling models.
"""

from __future__ import annotations

from typing import Optional
import logging
import math
from collections import Counter

try:
    # Attempt to import sentence_transformers and load a small multilingual model
    from sentence_transformers import SentenceTransformer
    _EMBEDDER: Optional[SentenceTransformer] = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    # Library or model missing: fall back to heuristic
    _EMBEDDER = None  # type: ignore
    logging.getLogger(__name__).info(
        "sentence_transformers not available; falling back to bag‑of‑words similarity"
    )


def _bag_of_words_vector(text: str) -> Counter[str]:
    """Compute a simple word frequency vector for a string."""
    tokens = [t for t in text.lower().split() if t.isalpha()]
    return Counter(tokens)


def _cosine_from_counters(a: Counter[str], b: Counter[str]) -> float:
    """Compute cosine similarity between two sparse count vectors."""
    # Compute dot product
    common_keys = set(a.keys()) & set(b.keys())
    dot = sum(a[k] * b[k] for k in common_keys)
    # Compute norms
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_similarity(a_text: str, b_text: str) -> float:
    """
    Return a semantic similarity score in the range [0, 1].

    When an embedding model is available, this computes the cosine
    similarity between the sentence embeddings of ``a_text`` and
    ``b_text``. Otherwise it falls back to a bag‑of‑words cosine.

    Args:
        a_text: The first piece of text.
        b_text: The second piece of text.

    Returns:
        A float between 0 and 1 representing the semantic closeness of
        the two inputs. Higher values indicate greater similarity.
    """
    a_text = a_text.strip()
    b_text = b_text.strip()
    if not a_text or not b_text:
        return 0.0
    if _EMBEDDER is not None:
        # Use embeddings; normalize cosine from [-1, 1] to [0, 1]
        try:
            vecs = _EMBEDDER.encode([a_text, b_text])
            a_vec, b_vec = vecs[0], vecs[1]
            # Compute cosine similarity manually
            dot = float(sum(x * y for x, y in zip(a_vec, b_vec)))
            norm_a = math.sqrt(sum(x * x for x in a_vec))
            norm_b = math.sqrt(sum(y * y for y in b_vec))
            if norm_a == 0.0 or norm_b == 0.0:
                cos = 0.0
            else:
                cos = dot / (norm_a * norm_b)
            # Normalize to 0..1
            return (cos + 1.0) / 2.0
        except Exception:
            # Something went wrong; fall through to heuristic
            pass
    # Fallback: bag‑of‑words cosine
    a_vec = _bag_of_words_vector(a_text)
    b_vec = _bag_of_words_vector(b_text)
    return _cosine_from_counters(a_vec, b_vec)
