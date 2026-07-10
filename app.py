"""
Streamlit demo UI for the hybrid RAG pipeline.

Upload documents, build the hybrid index, ask questions, and see the
answer alongside a per-sentence citation verification report.

Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud the same way you deployed your other projects.
Note: this app calls a local Ollama server by default for generation
(config.py -> GeneratorConfig.backend). On Streamlit Cloud there is no
local Ollama, so switch backend to "openai_compatible" and set an API
key in Streamlit secrets before deploying.
"""

import os
import tempfile

import streamlit as st

from config import DEFAULT_CONFIG
from src.pipeline import HybridRAGPipeline

st.set_page_config(page_title="Hybrid RAG Pipeline", page_icon="🔎", layout="wide")

st.title("🔎 Hybrid RAG Pipeline")
st.caption("Dense retrieval + BM25 + reranking + citation verification")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

with st.sidebar:
    st.header("1. Build the index")
    uploaded_files = st.file_uploader(
        "Upload .txt, .md, or .pdf files", accept_multiple_files=True
    )
    build_clicked = st.button("Build hybrid index", type="primary")

    st.divider()
    st.header("Generator backend")
    backend = st.selectbox("Backend", ["ollama", "openai_compatible"], index=0)
    DEFAULT_CONFIG.generator.backend = backend
    if backend == "ollama":
        DEFAULT_CONFIG.generator.ollama_model = st.text_input(
            "Ollama model", value=DEFAULT_CONFIG.generator.ollama_model
        )
    else:
        DEFAULT_CONFIG.generator.openai_compatible_model = st.text_input(
            "Model name", value=DEFAULT_CONFIG.generator.openai_compatible_model
        )
        st.caption("Set the API key via the OPENAI_API_KEY environment variable or Streamlit secrets.")

    st.divider()
    st.header("Citation verification")
    DEFAULT_CONFIG.citation.method = st.selectbox(
        "Method", ["embedding", "nli"], index=0,
        help="'embedding' is fast and needs no extra model. 'nli' is stricter but slower."
    )

if build_clicked:
    if not uploaded_files:
        st.sidebar.error("Upload at least one document first.")
    else:
        with st.spinner("Chunking, embedding, and indexing..."):
            tmp_dir = tempfile.mkdtemp()
            for f in uploaded_files:
                with open(os.path.join(tmp_dir, f.name), "wb") as out:
                    out.write(f.read())
            pipeline = HybridRAGPipeline(DEFAULT_CONFIG)
            num_chunks = pipeline.ingest_directory(tmp_dir)
            st.session_state.pipeline = pipeline
            st.session_state.num_chunks = num_chunks
        st.sidebar.success(f"Indexed {st.session_state.num_chunks} chunks from {len(uploaded_files)} file(s).")

st.header("2. Ask a question")
query = st.text_input("Your question", placeholder="e.g. How does hybrid search combine dense and BM25 retrieval?")
ask_clicked = st.button("Ask")

if ask_clicked:
    if st.session_state.pipeline is None:
        st.error("Build the index first (see sidebar).")
    elif not query.strip():
        st.error("Enter a question.")
    else:
        with st.spinner("Retrieving, reranking, and generating..."):
            try:
                response = st.session_state.pipeline.query(query)
            except Exception as e:
                st.error(f"Generation failed: {e}\n\nIf using the 'ollama' backend, make sure Ollama is running locally with the selected model pulled.")
                response = None

        if response:
            st.subheader("Answer")
            st.write(response.answer)

            st.subheader("Citation verification")
            summary = response.verification_summary
            if summary["checked_sentences"] == 0:
                st.info("No checkable sentences found in the answer.")
            else:
                rate = summary["support_rate"] or 0.0
                st.metric("Support rate", f"{rate * 100:.0f}%", help="Share of factual sentences backed by a retrieved chunk")
                for v in response.verdicts:
                    if v.verdict == "SUPPORTED":
                        st.success(f"✅ {v.sentence}  \n*Source: {v.best_source} (confidence {v.confidence:.2f})*")
                    elif v.verdict == "UNSUPPORTED":
                        st.warning(f"⚠️ {v.sentence}  \n*No source cleared the confidence threshold ({v.confidence:.2f})*")

            with st.expander("Retrieved chunks (after hybrid fusion + reranking)"):
                for i, c in enumerate(response.retrieved_chunks, 1):
                    st.markdown(f"**{i}. {c.source}** (chunk {c.chunk_index})")
                    st.text(c.text[:500] + ("..." if len(c.text) > 500 else ""))

st.divider()
st.caption(
    f"Index status: {'built (' + str(st.session_state.num_chunks) + ' chunks)' if st.session_state.pipeline else 'not built'}"
)
