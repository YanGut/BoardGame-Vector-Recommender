from __future__ import annotations

from typing import Iterable, List, Tuple

from haystack import Document


def hybrid_rank(
    documents: Iterable[Document],
    semantic_weight: float,
    popularity_weight: float,
    top_k: int,
) -> List[Tuple[Document, float]]:
    """
    Rank the provided documents using a weighted blend of semantic and popularity scores.
    """
    ranked_results: List[Tuple[Document, float]] = []

    for doc in documents:
        semantic_score = float(getattr(doc, "score", 0.0) or 0.0)
        popularity_score = float(doc.meta.get("popularity_score", 0.0) or 0.0)
        final_score = (semantic_weight * semantic_score) + (popularity_weight * popularity_score)
        ranked_results.append((doc, final_score))

    ranked_results.sort(key=lambda item: item[1], reverse=True)
    return ranked_results[:top_k]
