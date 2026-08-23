"""
chunking.py
Loads the markdown knowledge base documents and splits them into
retrieval-sized chunks.

Design choice: chunk along document structure (## sections), not a blind
fixed-token sliding window. Every document in this knowledge base was
hand-organized into ## sections that are each already a coherent,
self-contained unit (one clinical concept per section) — splitting there
preserves meaning at chunk boundaries, which a fixed-window splitter would
sometimes cut through the middle of a sentence or idea. Sections that run
long are further split by paragraph so no single chunk is large enough to
dilute embedding relevance.
"""

import re
from dataclasses import dataclass
from pathlib import Path

KB_DIR = Path("data/knowledge_base")
MAX_CHUNK_WORDS = 180  # sections longer than this get split further
FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section_title: str
    text: str
    source: str
    title: str
    url: str
    retrieved: str


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    match = FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    fm_block, body = match.group(1), match.group(2)
    meta = {}
    for line in fm_block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, body


def _split_into_sections(body: str) -> list[tuple[str, str]]:
    """Split on '## ' headers. Text under the '# ' H1 title (before the
    first '## ') is folded into the first section rather than dropped."""
    lines = body.strip().splitlines()
    sections: list[tuple[str, str]] = []
    current_title = None
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("## "):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = []
        elif line.startswith("# "):
            current_title = line[2:].strip()
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [(t, txt) for t, txt in sections if txt]


def _split_long_section(text: str, max_words: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current, current_words = [], [], 0
    for para in paragraphs:
        para_words = len(para.split())
        if current and current_words + para_words > max_words:
            chunks.append("\n\n".join(current))
            current, current_words = [], 0
        current.append(para)
        current_words += para_words
    if current:
        chunks.append("\n\n".join(current))
    return chunks if chunks else [text]


def load_and_chunk_documents(kb_dir: Path = KB_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    md_files = sorted(
        f for f in kb_dir.glob("*.md") if f.name != "sources_manifest.md"
    )

    for path in md_files:
        doc_id = path.stem
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw)
        sections = _split_into_sections(body)

        for section_idx, (section_title, section_text) in enumerate(sections):
            sub_chunks = _split_long_section(section_text, MAX_CHUNK_WORDS)
            for sub_idx, sub_text in enumerate(sub_chunks):
                chunk_id = f"{doc_id}::{section_idx}.{sub_idx}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        section_title=section_title or meta.get("title", doc_id),
                        text=sub_text,
                        source=meta.get("source", "unknown"),
                        title=meta.get("title", doc_id),
                        url=meta.get("url", ""),
                        retrieved=meta.get("retrieved", ""),
                    )
                )
    return chunks


if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    word_counts = [len(c.text.split()) for c in chunks]
    print(f"{len(chunks)} chunks from {len(set(c.doc_id for c in chunks))} documents")
    print(
        f"chunk word count: min={min(word_counts)} max={max(word_counts)} "
        f"avg={sum(word_counts)/len(word_counts):.0f}"
    )
    for c in chunks[:3]:
        print(f"\n--- {c.chunk_id} ({c.section_title}) ---")
        print(c.text[:200])
