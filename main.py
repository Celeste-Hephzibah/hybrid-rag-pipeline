"""
Command-line entry point.

Build an index:
    python main.py build --docs data/sample_docs --index index_store

Query an existing index:
    python main.py ask --index index_store --q "What is hybrid search?"

Build and immediately ask (no persistence, quick testing):
    python main.py demo --docs data/sample_docs --q "What is FAISS used for?"
"""

import argparse

from config import DEFAULT_CONFIG
from src.pipeline import HybridRAGPipeline


def cmd_build(args):
    pipeline = HybridRAGPipeline(DEFAULT_CONFIG)
    n = pipeline.ingest_directory(args.docs)
    pipeline.save_index(args.index)
    print(f"Indexed {n} chunks from '{args.docs}' -> saved to '{args.index}'")


def cmd_ask(args):
    pipeline = HybridRAGPipeline(DEFAULT_CONFIG)
    pipeline.load_index(args.index)
    _run_query(pipeline, args.q)


def cmd_demo(args):
    pipeline = HybridRAGPipeline(DEFAULT_CONFIG)
    n = pipeline.ingest_directory(args.docs)
    print(f"Indexed {n} chunks from '{args.docs}' (in-memory, not saved)")
    _run_query(pipeline, args.q)


def _run_query(pipeline, question):
    response = pipeline.query(question)
    print("\n" + "=" * 70)
    print("ANSWER:")
    print(response.answer)
    print("\n" + "-" * 70)
    print("CITATION VERIFICATION:")
    summary = response.verification_summary
    if summary["checked_sentences"] == 0:
        print("  (no checkable sentences)")
    else:
        print(f"  Support rate: {summary['support_rate']*100:.0f}% "
              f"({summary['supported_sentences']}/{summary['checked_sentences']} sentences)")
        for v in response.verdicts:
            tag = {"SUPPORTED": "[OK]", "UNSUPPORTED": "[!!]", "SKIPPED": "[--]"}[v.verdict]
            src = f" -> {v.best_source}" if v.best_source else ""
            print(f"  {tag} {v.sentence}{src}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid RAG pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build and save an index from a directory of documents")
    p_build.add_argument("--docs", required=True, help="Directory of .txt/.md/.pdf files")
    p_build.add_argument("--index", default=DEFAULT_CONFIG.index_dir, help="Directory to save the index to")
    p_build.set_defaults(func=cmd_build)

    p_ask = sub.add_parser("ask", help="Query a previously built index")
    p_ask.add_argument("--index", default=DEFAULT_CONFIG.index_dir, help="Directory the index was saved to")
    p_ask.add_argument("--q", required=True, help="Question to ask")
    p_ask.set_defaults(func=cmd_ask)

    p_demo = sub.add_parser("demo", help="Build an in-memory index and immediately ask one question")
    p_demo.add_argument("--docs", required=True, help="Directory of .txt/.md/.pdf files")
    p_demo.add_argument("--q", required=True, help="Question to ask")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)
