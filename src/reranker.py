"""
Cross-encoder reranking.

Dense + BM25 fusion is fast but scores query and document independently
(bi-encoder style). A cross-encoder jointly attends over the
(query, chunk) pair, which is slower but substantially more accurate at
distinguishing "topically related" from "actually answers the question."
Only rerank the fused shortlist (~15 chunks), never the full corpus.
"""

from __future__ import annotations

from typing import List, Tuple

from sentence_transformers import CrossEncoder

from config import RerankConfig
from src.chunking import Chunk


class Reranker:
    def __init__(self, config: RerankConfig):
        self.config = config
        self.model = CrossEncoder(config.model_name) if config.enabled else None

    def rerank(
        self, query: str, candidates: List[Tuple[Chunk, float]]
    ) -> List[Tuple[Chunk, float]]:
        if not self.config.enabled or self.model is None or not candidates:
            return candidates[: self.config.top_k]

        pairs = [(query, chunk.text) for chunk, _ in candidates]
        scores = self.model.predict(pairs)
        reranked = sorted(
            zip([c for c, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(chunk, float(score)) for chunk, score in reranked[: self.config.top_k]]
