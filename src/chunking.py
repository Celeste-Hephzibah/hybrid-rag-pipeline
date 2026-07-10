"""
Document loading and chunking.

Splits raw documents into overlapping chunks small enough for embedding
and reranking, while keeping enough context for citation verification.
Each chunk carries metadata (source file, chunk index, char offsets) so
that later stages can cite back to the exact passage used.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from typing import List

from pypdf import PdfReader

from config import ChunkingConfig


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_document(path: str) -> str:
    """Load raw text from a .txt, .md, or .pdf file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext in (".txt", ".md"):
        return _read_txt(path)
    raise ValueError(f"Unsupported file type: {ext}")


def _split_into_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter; avoids an NLTK punkt download requirement.
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    source: str,
    config: ChunkingConfig,
) -> List[Chunk]:
    """
    Chunk text into overlapping windows measured in whitespace tokens.
    Chunking respects sentence boundaries where possible to avoid cutting
    a sentence in half, which would hurt citation verification later.
    """
    sentences = _split_into_sentences(text)
    chunks: List[Chunk] = []

    current_words: List[str] = []
    current_sentences: List[str] = []

    def flush(idx: int):
        joined = " ".join(current_sentences).strip()
        if len(joined.split()) >= config.min_chunk_size or idx == 0:
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=joined,
                    source=source,
                    chunk_index=idx,
                )
            )

    idx = 0
    for sentence in sentences:
        sentence_words = sentence.split()
        if len(current_words) + len(sentence_words) > config.chunk_size and current_words:
            flush(idx)
            idx += 1
            # carry overlap forward from the end of the previous chunk
            overlap_words = current_words[-config.chunk_overlap:] if config.chunk_overlap else []
            current_words = list(overlap_words)
            # reconstruct sentence-level overlap approximately by keeping last 1-2 sentences
            current_sentences = current_sentences[-2:] if len(current_sentences) > 1 else []
        current_words.extend(sentence_words)
        current_sentences.append(sentence)

    if current_sentences:
        flush(idx)

    return chunks


def chunk_directory(directory: str, config: ChunkingConfig) -> List[Chunk]:
    """Load and chunk every supported file in a directory."""
    all_chunks: List[Chunk] = []
    for fname in sorted(os.listdir(directory)):
        path = os.path.join(directory, fname)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".txt", ".md", ".pdf"):
            continue
        text = load_document(path)
        all_chunks.extend(chunk_text(text, source=fname, config=config))
    return all_chunks
