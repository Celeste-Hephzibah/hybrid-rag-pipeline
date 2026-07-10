"""
Sanity tests for the parts of the pipeline that don't require downloading
embedding/reranking models (chunking, BM25, RRF fusion). Run with:

    python -m tests.test_pipeline_logic

For a full end-to-end test including dense retrieval, reranking, and
citation verification, run pipeline.py directly once you have internet
access to download the sentence-transformers/cross-encoder models
(they are NOT downloadable from this sandbox, but will work on your
machine or Streamlit Cloud).
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ChunkingConfig, SparseRetrievalConfig, FusionConfig
from src.chunking import chunk_directory, chunk_text
from src.sparse_retriever import SparseRetriever
from src.hybrid_fusion import reciprocal_rank_fusion


def test_chunking():
    chunks = chunk_directory("data/sample_docs", ChunkingConfig())
    assert len(chunks) > 0, "Expected at least one chunk"
    for c in chunks:
        word_count = len(c.text.split())
        assert word_count > 0
        assert c.source in ("ml_basics.txt", "mlops_deployment.txt")
    print(f"[PASS] chunking: produced {len(chunks)} chunks across 2 documents")
    return chunks


def test_bm25(chunks):
    retriever = SparseRetriever(SparseRetrievalConfig())
    retriever.build(chunks)

    # This query uses exact terminology ("FAISS") that BM25 should nail
    results = retriever.search("What is FAISS used for?", top_k=3)
    assert len(results) > 0
    top_chunk, top_score = results[0]
    assert "faiss" in top_chunk.text.lower(), "Expected FAISS chunk to rank highly for exact keyword match"
    print(f"[PASS] BM25: top result correctly contains 'FAISS' (score={top_score:.3f})")

    # Acronym / exact term test: BM25 should also surface CNN reliably
    results2 = retriever.search("CNN convolutional neural networks", top_k=3)
    assert any("cnn" in c.text.lower() or "convolutional" in c.text.lower() for c, _ in results2)
    print("[PASS] BM25: correctly retrieves CNN-related chunk")


def test_rrf_fusion():
    # Build two fake ranked lists sharing one overlapping chunk to verify
    # that RRF boosts chunks appearing near the top of both lists.
    class FakeChunk:
        def __init__(self, cid):
            self.id = cid

    a, b, c, d = FakeChunk("a"), FakeChunk("b"), FakeChunk("c"), FakeChunk("d")

    dense_results = [(a, 0.9), (b, 0.8), (c, 0.7)]
    sparse_results = [(b, 15.0), (d, 12.0), (a, 8.0)]

    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_k=4)
    fused_ids = [chunk.id for chunk, _ in fused]

    # 'b' is rank 2 in dense and rank 1 in sparse -> should fuse to the top
    # 'a' is rank 1 in dense and rank 3 in sparse -> should also rank highly
    assert fused_ids[0] in ("a", "b"), f"Expected 'a' or 'b' to lead fusion, got {fused_ids}"
    assert set(fused_ids) == {"a", "b", "c", "d"}
    print(f"[PASS] RRF fusion: ranked order = {fused_ids}")


def test_chunk_overlap_respects_sentences():
    long_text = " ".join([f"This is sentence number {i} about topic X." for i in range(80)])
    chunks = chunk_text(long_text, source="synthetic.txt", config=ChunkingConfig(chunk_size=50, chunk_overlap=10, min_chunk_size=5))
    assert len(chunks) > 1, "Expected long text to split into multiple chunks"
    for c in chunks:
        assert c.text.strip().endswith("."), "Chunk should end on a sentence boundary"
    print(f"[PASS] chunk overlap: {len(chunks)} chunks, all end on sentence boundaries")


if __name__ == "__main__":
    chunks = test_chunking()
    test_bm25(chunks)
    test_rrf_fusion()
    test_chunk_overlap_respects_sentences()
    print("\nAll logic tests passed.")
