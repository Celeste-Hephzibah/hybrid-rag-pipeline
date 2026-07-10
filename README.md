# Hybrid RAG Pipeline

A Retrieval-Augmented Generation pipeline combining **dense vector search**,
**BM25 sparse search**, **cross-encoder reranking**, and **post-hoc citation
verification** — built as a step up from a standard single-retriever RAG
chatbot.

## Why hybrid + reranking + verification?

| Stage | Problem it solves |
|---|---|
| Dense retrieval (FAISS + sentence-transformers) | Catches semantically related passages even when wording differs from the query |
| BM25 sparse retrieval | Catches exact keyword/acronym/rare-term matches that embeddings tend to blur |
| Reciprocal Rank Fusion | Combines the two ranked lists without needing their raw scores to be comparable |
| Cross-encoder reranker | Re-scores the top ~15 fused candidates with a model that jointly attends over (query, chunk) pairs — much more accurate than bi-encoder similarity, but too slow to run over a full corpus |
| Citation verification | Checks each sentence of the generated answer against the retrieved chunks and flags anything unsupported — the actual guardrail against hallucination |

## Architecture

```
Documents (.txt/.md/.pdf)
        │
        ▼
   Chunking (sentence-aware, overlapping windows)
        │
        ├──────────────┬──────────────┐
        ▼              ▼              │
  Dense (FAISS)    BM25 (rank_bm25)   │
        │              │              │
        └──────┬───────┘              │
               ▼                      │
     Reciprocal Rank Fusion           │
               │                      │
               ▼                      │
     Cross-Encoder Reranker           │
               │                      │
               ▼                      │
   LLM Generation (Ollama / OpenAI-compatible) with inline [Source: ...] citations
               │
               ▼
   Citation Verifier (per-sentence support check against retrieved chunks)
               │
               ▼
        Answer + verification report
```

## Project structure

```
hybrid-rag-pipeline/
├── config.py                  # all tunable parameters in one place
├── main.py                    # CLI: build / ask / demo
├── app.py                     # Streamlit UI
├── src/
│   ├── chunking.py            # document loading + sentence-aware chunking
│   ├── dense_retriever.py     # sentence-transformers + FAISS
│   ├── sparse_retriever.py    # BM25 via rank_bm25
│   ├── hybrid_fusion.py       # RRF and weighted fusion
│   ├── reranker.py            # cross-encoder reranking
│   ├── citation_verifier.py   # per-sentence citation checking
│   ├── generator.py           # Ollama / OpenAI-compatible generation
│   └── pipeline.py            # orchestrates all of the above
├── data/sample_docs/          # 2 sample docs to test with immediately
├── tests/test_pipeline_logic.py
└── requirements.txt
```

## Setup

```bash
py -m venv venv
venv\Scripts\activate          # Windows, matches your existing setup
pip install -r requirements.txt
```

The first run will download the embedding model
(`sentence-transformers/all-MiniLM-L6-v2`, ~90MB) and the reranker
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~90MB) from Hugging Face —
this needs internet access once, then they're cached locally.

## Generation backend

By default this uses **Ollama + TinyLlama**, matching your existing local
setup from the RAG Document Chatbot project:

```bash
ollama pull tinyllama
ollama serve
```

To use OpenAI or another OpenAI-compatible endpoint instead, edit
`config.py`:

```python
GeneratorConfig(
    backend="openai_compatible",
    openai_compatible_url="https://api.openai.com/v1/chat/completions",
    openai_compatible_model="gpt-4o-mini",
)
```

and set the `OPENAI_API_KEY` environment variable.

## Usage

### CLI — quick test with the included sample docs

```bash
python main.py demo --docs data/sample_docs --q "How does hybrid search combine dense and BM25 retrieval?"
```

### CLI — build a persistent index, then query it later

```bash
python main.py build --docs data/sample_docs --index index_store
python main.py ask --index index_store --q "What is a cross-encoder reranker?"
```

### Streamlit app

```bash
streamlit run app.py
```

Upload documents in the sidebar, build the index, then ask questions.
Each answer shows a per-sentence citation verification report — green for
sentences backed by a retrieved chunk, yellow for anything the verifier
couldn't confirm.

## Local model choice matters: a debugging walkthrough

While testing this pipeline locally, generation quality varied a lot
depending on which local Ollama model was used — worth documenting
because it's a good illustration of how to isolate *where* in a RAG
stack a failure actually lives.

**1. TinyLlama (1.1B) conflated two different concepts.** Asked "how
does hybrid search combine dense and BM25 retrieval," it answered with
details about cross-encoder reranking instead — a different section of
the source document. The citation verifier caught this immediately
(33% support rate), correctly flagging the conflated sentences as
unsupported by the retrieved chunks.

**2. That surfaced a real bug in the verifier itself.** The original
implementation compared each answer sentence against a whole retrieved
chunk's embedding. When a chunk covers several topics, its aggregate
embedding gets diluted across all of them — so even a sentence quoted
verbatim from that chunk could score below the similarity threshold.
Fixed by comparing against individual *sentences* within each chunk
instead of the chunk as a whole; support rates on correct answers went
from ~40% to 90%+ after the fix.

**3. TinyLlama still ignored explicit prompt instructions.** Even after
adding "answer in ≤4 sentences," "paraphrase, don't copy verbatim," and
a hard `num_predict` token cap at the Ollama API level, TinyLlama kept
echoing the entire retrieved context back almost word-for-word,
truncated mid-sentence at the token limit. This is a model capacity
ceiling, not a prompt or pipeline issue — a 1.1B model is simply too
small to reliably follow multi-part instructions.

**4. Swapping to Llama 3.2 (3B) fixed it with zero other pipeline
changes.** Same retrieval, same fusion, same reranker, same verifier —
just a different `ollama_model` string in `config.py`. The result was a
concise, correctly synthesized, 100%-cited answer.

**Takeaway:** retrieval, fusion, reranking, and citation verification
all worked correctly throughout — the failure was isolated entirely to
generation, and specifically to model capacity rather than prompting.
Being able to point to *which stage* broke (and prove it with the
verifier's numbers) is the actual value of building citation
verification into the pipeline rather than treating the LLM as a black
box.

## Deploying to Streamlit Cloud

Same pattern as your other Streamlit Cloud deployments. Two things to
change before deploying:

1. Switch `GeneratorConfig.backend` to `"openai_compatible"` — there's no
   local Ollama server on Streamlit Cloud — and add your API key to
   Streamlit's secrets manager.
2. If your document corpus is large, embedding + FAISS index files can get
   big; use Git LFS for `index_store/` the same way you did for the
   Twitter Sentiment Analyzer's model file.

## Tuning

Every retrieval/reranking/verification parameter lives in `config.py`:
chunk size and overlap, how many candidates each retriever pulls before
fusion, RRF's `k` constant vs. weighted fusion, how many chunks survive
reranking, and the similarity threshold the citation verifier uses to
mark a sentence as supported. Start by adjusting `fused_top_k` and
`rerank.top_k` — those have the biggest effect on answer quality vs.
generation cost.

## Citation verification: two modes

- `"embedding"` (default) — cosine similarity between each answer
  sentence and the retrieved chunks, using the same embedding model as
  dense retrieval. No extra download, fast enough to run on every query.
- `"nli"` — a natural-language-inference cross-encoder checks whether a
  source chunk actually *entails* the sentence, not just whether it's
  topically similar. Stricter, catches subtler unsupported claims, but
  slower and needs an extra model download
  (`cross-encoder/nli-deberta-v3-small`).

Switch via `config.py` → `CitationVerificationConfig.method`.

## Known limitations (be upfront about these in interviews)

- The embedding-based citation check catches topical mismatch and outright
  fabrication well, but can miss cases where a sentence is topically
  similar to a source yet subtly misstates what it says — the `"nli"`
  mode is the fix for that class of error, at a speed cost.
- BM25 and the dense index are both rebuilt from scratch on `ingest_*` —
  there's no incremental update path. For a corpus that changes
  frequently, add incremental indexing rather than full rebuilds.
- The sentence splitter is regex-based, not a full NLP sentence
  tokenizer, so it can occasionally mis-split on abbreviations
  (e.g. "Dr. Smith"). Fine for most prose, worth swapping for `nltk` or
  `spacy` sentence segmentation if your documents are heavy on
  abbreviations.
