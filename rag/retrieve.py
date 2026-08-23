"""
retrieve.py
Query-time retrieval against the persisted FAISS index. This is the exact
function the agent's Medical Knowledge Retrieval Tool will call in P4 —
built now as a standalone, independently testable module rather than
agent-embedded logic, per the project's tool-independence requirement.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from rag.build_index import EMBEDDING_MODEL_NAME, INDEX_PATH, METADATA_PATH

logger = logging.getLogger(__name__)

_model_cache = None


@dataclass
class RetrievedPassage:
    text: str
    section_title: str
    source: str
    title: str
    url: str
    retrieved: str
    score: float

    def citation(self) -> str:
        """Human-readable citation string for report/UI display — this is
        the exact format the Report Generator tool will embed."""
        return f"{self.title} ({self.source}), section \"{self.section_title}\" — {self.url}"


def _get_model() -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return _model_cache


def retrieve(query: str, k: int = 3, min_score: float = 0.25) -> list[RetrievedPassage]:
    """Retrieve the top-k most relevant knowledge base passages for a
    query. Returns an empty list (not fabricated content) if nothing
    clears min_score — the calling tool/agent must handle "no relevant
    evidence found" explicitly rather than always getting k results."""
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {INDEX_PATH}. Run `python -m rag.build_index` first."
        )

    index = faiss.read_index(str(INDEX_PATH))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    model = _get_model()
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    scores, ids = index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1 or score < min_score:
            continue
        m = metadata[idx]
        results.append(
            RetrievedPassage(
                text=m["text"],
                section_title=m["section_title"],
                source=m["source"],
                title=m["title"],
                url=m["url"],
                retrieved=m["retrieved"],
                score=float(score),
            )
        )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import sys

    query_text = " ".join(sys.argv[1:]) or "what does high cholesterol mean for heart disease risk"
    passages = retrieve(query_text, k=3)
    print(f"\nQuery: {query_text!r}\n")
    if not passages:
        print("No passages cleared the relevance threshold.")
    for p in passages:
        print(f"[{p.score:.3f}] {p.citation()}")
        print(f"  {p.text[:200]}...\n")
