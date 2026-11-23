"""Reciprocal Rank Fusion (alpha = 60) over BM25, vector and graph rank lists, plus a light reranker."""
from __future__ import annotations

from ..index.store import tokenize


def rrf(rank_lists: list[list[str]], alpha: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for lst in rank_lists:
        for rank, cid in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (alpha + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def rerank(query: str, fused: list[tuple[str, float]], texts: dict[str, str], top_k: int) -> list[str]:
    """Blend normalised RRF score with query-term coverage (no cross-encoder download needed)."""
    if not fused:
        return []
    qt = set(tokenize(query))
    top = fused[0][1] or 1.0
    scored = []
    for cid, s in fused:
        cov = len(qt & set(tokenize(texts[cid]))) / max(len(qt), 1)
        scored.append((0.6 * s / top + 0.4 * cov, cid))
    return [cid for _, cid in sorted(scored, reverse=True)[:top_k]]
