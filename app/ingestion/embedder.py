"""Singleton embedder using BAAI/bge-small-en-v1.5."""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embed_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    # bge models expect a query prefix for retrieval tasks
    return embed_texts([f"Represent this sentence for searching relevant passages: {query}"])[0]
