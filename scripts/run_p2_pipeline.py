"""One-shot driver for P2: build the FAISS index from the real knowledge
base and demo retrieval with citations on a handful of queries chosen to
map onto the ML model's actual features (cholesterol, chest pain, blood
pressure, ECG/stress test, mortality stats).

Needs internet access on first run only, to download the
sentence-transformers/all-MiniLM-L6-v2 embedding model (~80MB) from
HuggingFace — same one-time requirement as pulling the Ollama LLM. After
that first download the model is cached locally and this runs fully
offline, consistent with the project's offline constraint.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

from rag.build_index import build_index
from rag.retrieve import retrieve

DEMO_QUERIES = [
    "what does high LDL cholesterol mean for heart disease risk",
    "chest pain during exercise",
    "what is a normal blood pressure reading",
    "what does an exercise stress test measure",
    "how many people die from heart disease each year",
]


def main():
    try:
        index, metadata = build_index(save=True)
        print(f"\nIndex built: {index.ntotal} chunks from the real knowledge base.\n")

        for q in DEMO_QUERIES:
            results = retrieve(q, k=2)
            print(f"Query: {q!r}")
            if not results:
                print("  (no passage cleared the relevance threshold)")
            for r in results:
                print(f"  [{r.score:.3f}] {r.citation()}")
            print()

    except Exception as exc:
        print(
            "\nCould not build/query the index — this step needs internet access "
            "on first run to download the embedding model from HuggingFace.\n"
            f"Underlying error: {exc}"
        )


if __name__ == "__main__":
    main()
