"""
Citation verification: the guardrail against hallucination.

After the generator produces an answer, this module breaks the answer
into sentences and checks each one against the retrieved chunks it
claims to be grounded in. Two strategies are supported:

  - "embedding": cosine similarity between the answer sentence and each
    *sentence* of each source chunk (not the whole chunk at once) using
    the same embedding model as dense retrieval. Fast, no extra model
    download, good enough to catch outright fabrication and unsupported
    claims.

  - "nli": a cross-encoder trained for natural language inference
    (NLI) checks whether a source sentence *entails* the answer
    sentence, rather than just being topically similar. Stricter and
    slower, catches subtler cases (e.g. a sentence that's on-topic but
    contradicts or overstates what the source says).

Both strategies compare against individual source *sentences*, not
whole chunks. Comparing against a whole multi-topic chunk dilutes its
embedding across everything the chunk discusses, which can make even a
verbatim, correctly-cited sentence score below threshold. Sentence-level
comparison avoids that.

Every sentence in the final answer gets a verdict: SUPPORTED,
UNSUPPORTED, or SKIPPED (too short to meaningfully check, e.g. "Yes.").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import CitationVerificationConfig
from src.chunking import Chunk


@dataclass
class SentenceVerdict:
    sentence: str
    verdict: str  # "SUPPORTED" | "UNSUPPORTED" | "SKIPPED"
    best_source: str | None
    confidence: float


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class CitationVerifier:
    def __init__(
        self,
        config: CitationVerificationConfig,
        embedding_model: SentenceTransformer | None = None,
    ):
        self.config = config
        # Reuse the dense retriever's embedding model when available to
        # avoid loading a second model into memory.
        self.embedding_model = embedding_model or SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.nli_model = (
            CrossEncoder(config.nli_model_name) if config.method == "nli" else None
        )

    @staticmethod
    def _build_source_sentence_pool(sources: List[Chunk]) -> tuple[List[str], List[str]]:
        """
        Flatten every source chunk into individual sentences, keeping a
        parallel list of which chunk's `source` filename each sentence
        came from. This is what verification actually compares against,
        instead of the whole chunk's averaged-out embedding.
        """
        pool_sentences: List[str] = []
        pool_sources: List[str] = []
        for chunk in sources:
            for sent in _split_sentences(chunk.text):
                pool_sentences.append(sent)
                pool_sources.append(chunk.source)
        return pool_sentences, pool_sources

    def _verify_embedding(
        self,
        sentence: str,
        pool_sentences: List[str],
        pool_sources: List[str],
        pool_embs: np.ndarray,
    ) -> tuple[str | None, float]:
        sent_emb = self.embedding_model.encode([sentence], normalize_embeddings=True)
        sims = pool_embs @ sent_emb.T
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx][0])
        source_name = (
            pool_sources[best_idx]
            if best_score >= self.config.embedding_similarity_threshold
            else None
        )
        return source_name, best_score

    def _verify_nli(
        self, sentence: str, pool_sentences: List[str], pool_sources: List[str]
    ) -> tuple[str | None, float]:
        # NLI cross-encoders expect (premise, hypothesis) pairs and return
        # a score representing how well the premise entails the hypothesis.
        # Using individual source sentences as premises (instead of whole
        # chunks) keeps each comparison focused on one claim at a time.
        pairs = [(src_sent, sentence) for src_sent in pool_sentences]
        scores = self.nli_model.predict(pairs)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        source_name = (
            pool_sources[best_idx]
            if best_score >= self.config.nli_entailment_threshold
            else None
        )
        return source_name, best_score

    def verify(self, answer: str, sources: List[Chunk]) -> List[SentenceVerdict]:
        verdicts = []
        answer_sentences = _split_sentences(answer)

        if not sources or not answer_sentences:
            return [SentenceVerdict(s, "SKIPPED", None, 0.0) for s in answer_sentences]

        pool_sentences, pool_sources = self._build_source_sentence_pool(sources)
        if not pool_sentences:
            return [SentenceVerdict(s, "SKIPPED", None, 0.0) for s in answer_sentences]

        # Pre-embed the source sentence pool once per call (not once per
        # answer sentence) since it's identical across the whole answer.
        pool_embs = None
        if self.config.method != "nli":
            pool_embs = self.embedding_model.encode(pool_sentences, normalize_embeddings=True)

        for sentence in answer_sentences:
            if len(sentence) < self.config.sentence_min_length:
                verdicts.append(SentenceVerdict(sentence, "SKIPPED", None, 0.0))
                continue

            if self.config.method == "nli":
                source_name, score = self._verify_nli(sentence, pool_sentences, pool_sources)
            else:
                source_name, score = self._verify_embedding(
                    sentence, pool_sentences, pool_sources, pool_embs
                )

            verdict = "SUPPORTED" if source_name else "UNSUPPORTED"
            verdicts.append(SentenceVerdict(sentence, verdict, source_name, score))
        return verdicts

    @staticmethod
    def summarize(verdicts: List[SentenceVerdict]) -> dict:
        checked = [v for v in verdicts if v.verdict != "SKIPPED"]
        supported = [v for v in checked if v.verdict == "SUPPORTED"]
        return {
            "total_sentences": len(verdicts),
            "checked_sentences": len(checked),
            "supported_sentences": len(supported),
            "support_rate": (len(supported) / len(checked)) if checked else None,
            "unsupported": [v.sentence for v in checked if v.verdict == "UNSUPPORTED"],
        }
