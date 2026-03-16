"""Hierarchical text chunker: parent chunks + child chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    content: str
    metadata: dict = field(default_factory=dict)
    parent_index: int | None = None


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _window(sentences: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    step = max(1, size - overlap)
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i : i + size])
        if chunk:
            chunks.append(chunk)
    return chunks


def hierarchical_chunk(
    text: str,
    doc_id: str,
    parent_size: int = 10,
    child_size: int = 3,
    overlap: int = 1,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Returns both parent and child chunks.
    Child chunks store a reference to their parent index.
    """
    meta = metadata or {}
    sentences = _split_sentences(text)
    chunks: list[Chunk] = []
    index = 0

    parent_windows = _window(sentences, parent_size, overlap)
    for p_idx, parent_text in enumerate(parent_windows):
        parent_chunk = Chunk(
            doc_id=doc_id,
            chunk_index=index,
            content=parent_text,
            metadata={**meta, "level": "parent"},
        )
        parent_index = index
        chunks.append(parent_chunk)
        index += 1

        child_sentences = _split_sentences(parent_text)
        child_windows = _window(child_sentences, child_size, overlap)
        for child_text in child_windows:
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_index=index,
                    content=child_text,
                    metadata={**meta, "level": "child"},
                    parent_index=parent_index,
                )
            )
            index += 1

    return chunks
