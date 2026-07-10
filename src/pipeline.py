"""
End-to-end orchestration: ingest -> hybrid retrieve -> fuse -> rerank ->
generate -> verify citations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from config import PipelineConfig, DEFAULT_CONFIG
from src.chunking import Chunk, chunk_directory
from src.dense_retriever import DenseRetriever
from src.sparse_retriever import SparseRetriever
from src.hybrid_fusion import fuse
from src.reranker import Reranker
from src.citation_verifier import CitationVerifier, SentenceVerdict
from src.generator import generate_answer


@dataclass
class RAGResponse:
    query: str
    answer: str
    retrieved_chunks: List[Chunk]
    verdicts: List[SentenceVerdict]
    verification_summary: dict
    debug: dict = field(default_factory=dict)


class HybridRAGPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self.dense = DenseRetriever(self.config.dense)
        self.sparse = SparseRetriever(self.config.sparse)
        self.reranker = Reranker(self.config.rerank)
        # Reuse the dense retriever's embedding model for citation checks
        self.verifier = CitationVerifier(
            self.config.citation, embedding_model=self.dense.model
        )
        self._built = False

    # ---------- Indexing ----------

    def ingest_directory(self, directory: str) -> int:
        chunks = chunk_directory(directory, self.config.chunking)
        if not chunks:
            raise ValueError(f"No supported documents found in {directory}")
        self.dense.build(chunks)
        self.sparse.build(chunks)
        self._built = True
        return len(chunks)

    def ingest_chunks(self, chunks: List[Chunk]) -> int:
        self.dense.build(chunks)
        self.sparse.build(chunks)
        self._built = True
        return len(chunks)

    def save_index(self, directory: str | None = None) -> None:
        directory = directory or self.config.index_dir
        self.dense.save(directory)
        self.sparse.save(directory)

    def load_index(self, directory: str | None = None) -> None:
        directory = directory or self.config.index_dir
        self.dense.load(directory)
        self.sparse.load(directory)
        self._built = True

    # ---------- Querying ----------

    def retrieve(self, query: str) -> List[Chunk]:
        if not self._built:
            raise RuntimeError("No index loaded. Call ingest_directory() or load_index() first.")

        dense_results = self.dense.search(query, top_k=self.config.dense.top_k)
        sparse_results = self.sparse.search(query, top_k=self.config.sparse.top_k)
        fused = fuse(dense_results, sparse_results, self.config.fusion)
        reranked = self.reranker.rerank(query, fused)
        return [chunk for chunk, _ in reranked]

    def query(self, query: str) -> RAGResponse:
        top_chunks = self.retrieve(query)
        answer = generate_answer(query, top_chunks, self.config.generator)
        verdicts = self.verifier.verify(answer, top_chunks)
        summary = CitationVerifier.summarize(verdicts)

        return RAGResponse(
            query=query,
            answer=answer,
            retrieved_chunks=top_chunks,
            verdicts=verdicts,
            verification_summary=summary,
        )
