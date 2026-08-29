"""
build_index.py
Embeds every chunk with sentence-transformers and builds a FAISS index.

Design choices (see docs/architecture.md Section 5.4-5.5 for full
rationale):
- Embedding model: all-MiniLM-L6-v2, run on CPU. ~80MB, leaves the full
  GPU VRAM budget free for the local LLM (Qwen2.5-7B), which is the more
  VRAM-hungry component.
- FAISS index type: IndexFlatIP (exact inner-product search) over
  L2-normalized vectors, which makes inner product equivalent to cosine
  similarity. Flat/exact search is deliberate, not a placeholder — this
  corpus is 43 chunks. An approximate index (IVF, HNSW) only pays off at
  a scale where exact search becomes slow, and introduces recall/accuracy
  trade-offs this project doesn't need to accept for no benefit.
- Metadata (chunk text, source, url, section) is stored in a parallel JSON
  file, keyed by the same integer id FAISS uses internally, since FAISS
  itself only stores vectors.
"""

import json
import logging
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from rag.chunking import load_and_chunk_documents

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = Path("embeddings/faiss_index")
INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "chunks_metadata.json"


def build_index(save: bool = True):
    chunks = load_and_chunk_documents()
    logger.info("Loaded %d chunks from knowledge base", len(chunks))

    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    texts = [c.text for c in chunks]
    embeddings = model.encode(
        texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("Built FAISS IndexFlatIP: %d vectors, dim=%d", index.ntotal, dim)

    metadata = [
        {
            "id": i,
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "section_title": c.section_title,
            "text": c.text,
            "source": c.source,
            "title": c.title,
            "url": c.url,
            "retrieved": c.retrieved,
        }
        for i, c in enumerate(chunks)
    ]

    if save:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(INDEX_PATH))
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Saved index to %s and metadata to %s", INDEX_PATH, METADATA_PATH)

    return index, metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_index()
