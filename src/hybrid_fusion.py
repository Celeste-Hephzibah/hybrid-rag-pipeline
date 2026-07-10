"""
Fuses dense and sparse retrieval results into a single ranked list.

Default method: Reciprocal Rank Fusion (RRF). RRF is the standard choice
for hybrid search because it combines ranked lists without needing the
two systems' raw scores to be on comparable scales (cosine similarity
vs. BM25 score are not directly comparable; rank position is).

score(chunk) = sum over each retriever of 1 / (k + rank_in_that_retriever)
"""

from __future__ import annotations

from typing import List, Tuple

from config import FusionConfig
from src.chunking import Chunk


def reciprocal_rank_fusion(
    dense_results: List[Tuple[Chunk, float]],
    sparse_results: List[Tuple[Chunk, float]],
    k: int = 60,
    top_k: int = 15,
) -> List[Tuple[Chunk, float]]:
    scores: dict[str, float] = {}
    chunk_lookup: dict[str, Chunk] = {}

    for rank, (chunk, _) in enumerate(dense_results):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
        chunk_lookup[chunk.id] = chunk

    for rank, (chunk, _) in enumerate(sparse_results):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
        chunk_lookup[chunk.id] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(chunk_lookup[cid], score) for cid, score in ranked]


def weighted_score_fusion(
    dense_results: List[Tuple[Chunk, float]],
    sparse_results: List[Tuple[Chunk, float]],
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    top_k: int = 15,
) -> List[Tuple[Chunk, float]]:
    """
    Alternative to RRF: min-max normalizes each retriever's scores to
    [0, 1] then combines with fixed weights. More sensitive to tuning
    but can outperform RRF when one retriever is reliably stronger.
    """

    def normalize(results: List[Tuple[Chunk, float]]) -> dict[str, float]:
        if not results:
            return {}
        vals = [s for _, s in results]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        return {c.id: (s - lo) / span for c, s in results}

    dense_norm = normalize(dense_results)
    sparse_norm = normalize(sparse_results)
    chunk_lookup = {c.id: c for c, _ in dense_results + sparse_results}

    all_ids = set(dense_norm) | set(sparse_norm)
    combined = {
        cid: dense_weight * dense_norm.get(cid, 0.0)
        + sparse_weight * sparse_norm.get(cid, 0.0)
        for cid in all_ids
    }
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(chunk_lookup[cid], score) for cid, score in ranked]


def fuse(
    dense_results: List[Tuple[Chunk, float]],
    sparse_results: List[Tuple[Chunk, float]],
    config: FusionConfig,
) -> List[Tuple[Chunk, float]]:
    if config.method == "rrf":
        return reciprocal_rank_fusion(
            dense_results, sparse_results, k=config.rrf_k, top_k=config.fused_top_k
        )
    elif config.method == "weighted":
        return weighted_score_fusion(
            dense_results,
            sparse_results,
            dense_weight=config.dense_weight,
            sparse_weight=config.sparse_weight,
            top_k=config.fused_top_k,
        )
    raise ValueError(f"Unknown fusion method: {config.method}")
