"""
Dense retrieval: embeds chunks with a sentence-transformer model and
indexes them in FAISS for fast approximate nearest-neighbor search.
"""

from __future__ import annotations

import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import DenseRetrievalConfig
from src.chunking import Chunk


class DenseRetriever:
    def __init__(self, config: DenseRetrievalConfig):
        self.config = config
        self.model = SentenceTransformer(config.model_name)
        self.index: faiss.Index | None = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        # Inner product on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int | None = None) -> List[Tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("Dense index has not been built. Call build() first.")
        top_k = top_k or self.config.top_k
        query_emb = self.model.encode(
            [query],
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        ).astype("float32")
        scores, indices = self.index.search(query_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "dense.index"))
        with open(os.path.join(directory, "dense_chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, directory: str) -> None:
        self.index = faiss.read_index(os.path.join(directory, "dense.index"))
        with open(os.path.join(directory, "dense_chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
