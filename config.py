"""
Central configuration for the hybrid RAG pipeline.
Edit these values to tune retrieval, reranking, and generation behavior.
"""

from dataclasses import dataclass, field


@dataclass
class ChunkingConfig:
    chunk_size: int = 400          # tokens (approx, whitespace-split)
    chunk_overlap: int = 80        # tokens of overlap between consecutive chunks
    min_chunk_size: int = 50       # discard chunks smaller than this


@dataclass
class DenseRetrievalConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 20                # candidates pulled from dense index
    normalize_embeddings: bool = True


@dataclass
class SparseRetrievalConfig:
    top_k: int = 20                # candidates pulled from BM25
    k1: float = 1.5
    b: float = 0.75


@dataclass
class FusionConfig:
    method: str = "rrf"            # "rrf" (reciprocal rank fusion) or "weighted"
    rrf_k: int = 60                # standard RRF constant
    dense_weight: float = 0.5      # only used if method == "weighted"
    sparse_weight: float = 0.5     # only used if method == "weighted"
    fused_top_k: int = 15          # candidates passed to reranker


@dataclass
class RerankConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5                 # final chunks passed to the generator
    enabled: bool = True


@dataclass
class CitationVerificationConfig:
    method: str = "embedding"      # "embedding" (default, no extra model) or "nli"
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small"
    embedding_similarity_threshold: float = 0.55
    nli_entailment_threshold: float = 0.5
    sentence_min_length: int = 15  # skip trivially short sentences (e.g. "Yes.")


@dataclass
class GeneratorConfig:
    backend: str = "ollama"        # "ollama" or "openai_compatible"
    ollama_model: str = "llama3.2"  # tested: correctly synthesizes + cites. tinyllama works but is prone to context-echoing on small models -- see README.
    ollama_url: str = "http://localhost:11434/api/generate"
    openai_compatible_url: str = "http://localhost:8000/v1/chat/completions"
    openai_compatible_model: str = "gpt-4o-mini"
    openai_api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 220


@dataclass
class PipelineConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    dense: DenseRetrievalConfig = field(default_factory=DenseRetrievalConfig)
    sparse: SparseRetrievalConfig = field(default_factory=SparseRetrievalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    citation: CitationVerificationConfig = field(default_factory=CitationVerificationConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    index_dir: str = "index_store"


DEFAULT_CONFIG = PipelineConfig()
