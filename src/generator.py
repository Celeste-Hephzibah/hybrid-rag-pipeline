"""
Answer generation with citation-aware prompting.

Supports two backends:
  - "ollama": local inference (matches your existing TinyLlama/Ollama setup)
  - "openai_compatible": any OpenAI-compatible chat completions endpoint
    (OpenAI itself, vLLM, LM Studio, etc.)

The prompt explicitly instructs the model to only use the provided
chunks and to tag claims with [Source: filename] so the citation
verifier has something concrete to check against.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import requests

from config import GeneratorConfig
from src.chunking import Chunk

SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question "
    "using ONLY the information in the provided source excerpts. "
    "For every factual claim, cite the source it came from in the form "
    "[Source: <filename>]. If the excerpts do not contain enough "
    "information to answer, say so explicitly instead of guessing.\n\n"
    "Strict formatting rules:\n"
    "- Answer in at most 4 sentences. Do not exceed this.\n"
    "- Directly answer the question first; do not restate the question "
    "or describe what you are about to do.\n"
    "- Do NOT copy sentences verbatim from the source excerpts. "
    "Synthesize and paraphrase in your own words instead.\n"
    "- Only include information relevant to the question. Ignore "
    "source content that is topically unrelated, even if it appears "
    "in the excerpts provided."
)


def _build_context(chunks: List[Chunk]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[Source: {c.source} | chunk {c.chunk_index}]\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def _build_prompt(query: str, chunks: List[Chunk]) -> str:
    context = _build_context(chunks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"SOURCE EXCERPTS:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER (cite sources inline as [Source: filename]):"
    )


def _generate_ollama(prompt: str, config: GeneratorConfig) -> str:
    response = requests.post(
        config.ollama_url,
        json={
            "model": config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def _generate_openai_compatible(prompt: str, config: GeneratorConfig) -> str:
    api_key = os.environ.get(config.openai_api_key_env, "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        config.openai_compatible_url,
        headers=headers,
        json={
            "model": config.openai_compatible_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_answer(
    query: str, chunks: List[Chunk], config: GeneratorConfig
) -> str:
    prompt = _build_prompt(query, chunks)
    if config.backend == "ollama":
        return _generate_ollama(prompt, config)
    elif config.backend == "openai_compatible":
        return _generate_openai_compatible(prompt, config)
    raise ValueError(f"Unknown generator backend: {config.backend}")
