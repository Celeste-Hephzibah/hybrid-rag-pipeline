"""
Sparse retrieval: classic BM25 over tokenized chunk text.
Dense retrieval catches semantic/paraphrase matches; BM25 catches exact
keyword, acronym, and rare-term matches that embeddings tend to blur.
Combining both is the entire point of "hybrid" search.
"""

from __future__ import annotations

import os
import pickle
import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from config import SparseRetrievalConfig
from src.chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class SparseRetriever:
    def __init__(self, config: SparseRetrievalConfig):
        self.config = config
        self.bm25: BM25Okapi | None = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(
            tokenized_corpus, k1=self.config.k1, b=self.config.b
        )

    def search(self, query: str, top_k: int | None = None) -> List[Tuple[Chunk, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been built. Call build() first.")
        top_k = top_k or self.config.top_k
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_indices]

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "bm25.pkl"), "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)

    def load(self, directory: str) -> None:
        with open(os.path.join(directory, "bm25.pkl"), "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
