"""
Unit tests for rag/chunking.py, rag/build_index.py, rag/retrieve.py.

The chunking tests run anywhere, no network needed. The embedding/FAISS
tests need to download sentence-transformers/all-MiniLM-L6-v2 from
HuggingFace on first run — they're skipped automatically (not failed) if
that's unreachable, since that's an environment property, not a bug. On a
normal machine with internet access (required once during setup, same as
the Ollama model pull), these run for real.

Run with: pytest tests/unit/test_rag.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag.chunking import load_and_chunk_documents, _split_into_sections, _split_long_section


def _embedding_model_available() -> bool:
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        return True
    except Exception:
        return False


EMBEDDING_AVAILABLE = _embedding_model_available()
skip_if_no_embedding_model = pytest.mark.skipif(
    not EMBEDDING_AVAILABLE,
    reason="sentence-transformers model not reachable/cached in this environment",
)


# ---------- chunking.py (no network needed) ----------

def test_load_and_chunk_documents_produces_chunks():
    chunks = load_and_chunk_documents()
    assert len(chunks) > 0


def test_every_chunk_has_required_metadata():
    chunks = load_and_chunk_documents()
    for c in chunks:
        assert c.text.strip()
        assert c.url.startswith("http"), f"{c.chunk_id} missing a source URL"
        assert c.title
        assert c.source


def test_chunks_cover_all_knowledge_base_files():
    chunks = load_and_chunk_documents()
    doc_ids = {c.doc_id for c in chunks}
    kb_dir = Path("data/knowledge_base")
    expected = {
        f.stem for f in kb_dir.glob("*.md") if f.name != "sources_manifest.md"
    }
    assert doc_ids == expected


def test_no_chunk_wildly_exceeds_target_size():
    # Some slack above MAX_CHUNK_WORDS is fine (single long paragraphs
    # aren't split mid-paragraph), but nothing should be enormous.
    chunks = load_and_chunk_documents()
    for c in chunks:
        assert len(c.text.split()) < 400, f"{c.chunk_id} is unexpectedly large"


def test_split_into_sections_handles_headers():
    body = "# Title\nlead-in text\n## Section A\ntext a\n## Section B\ntext b"
    sections = _split_into_sections(body)
    titles = [t for t, _ in sections]
    assert "Section A" in titles
    assert "Section B" in titles


def test_split_long_section_respects_word_budget():
    long_text = "\n\n".join([f"Paragraph {i} with some words in it." for i in range(20)])
    parts = _split_long_section(long_text, max_words=20)
    assert len(parts) > 1
    for p in parts:
        assert len(p.split()) <= 40  # budget + one paragraph's worth of slack


# ---------- build_index.py / retrieve.py (need the embedding model) ----------

@skip_if_no_embedding_model
def test_build_index_creates_index_and_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import shutil

    shutil.copytree(Path(__file__).resolve().parents[2] / "data", tmp_path / "data")
    from rag.build_index import build_index, INDEX_PATH, METADATA_PATH

    index, metadata = build_index(save=True)
    assert index.ntotal == len(metadata)
    assert INDEX_PATH.exists()
    assert METADATA_PATH.exists()


@skip_if_no_embedding_model
def test_retrieve_returns_relevant_passage_for_cholesterol_query():
    from rag.build_index import build_index
    from rag.retrieve import retrieve

    build_index(save=True)  # ensure index exists in the real repo location
    results = retrieve("what is LDL and HDL cholesterol", k=3)
    assert len(results) > 0
    assert any("cholesterol" in r.text.lower() for r in results)


@skip_if_no_embedding_model
def test_retrieve_returns_relevant_passage_for_thal_query():
    """Regression test for a real gap found on a P4 hardware run: the top
    SHAP feature is very often `thal`, but the knowledge base originally
    had no document explaining what a reversible/fixed defect finding
    means — retrieval silently fell back to an unrelated blood-pressure
    passage instead of returning nothing or something relevant. This
    checks the fix (medlineplus_nuclear_stress_test.md) actually surfaces
    for the same query the agent builds via
    tools.knowledge_retrieval.build_query for a thal-driven prediction."""
    from rag.build_index import build_index
    from rag.retrieve import retrieve

    build_index(save=True)
    results = retrieve("thalassemia heart test result", k=2)
    assert len(results) > 0
    assert any("reversible" in r.text.lower() or "reversible" in r.title.lower() for r in results)


@skip_if_no_embedding_model
def test_retrieve_returns_empty_list_not_garbage_for_irrelevant_query():
    from rag.build_index import build_index
    from rag.retrieve import retrieve

    build_index(save=True)
    results = retrieve("best pizza toppings in Amman", k=3, min_score=0.6)
    # an unrelated query should not confidently match anything in a
    # cardiology knowledge base
    assert len(results) == 0


@skip_if_no_embedding_model
def test_citation_format_includes_source_and_url():
    from rag.build_index import build_index
    from rag.retrieve import retrieve

    build_index(save=True)
    results = retrieve("blood pressure categories", k=1)
    assert len(results) >= 1
    citation = results[0].citation()
    assert "http" in citation
    assert results[0].source in citation
