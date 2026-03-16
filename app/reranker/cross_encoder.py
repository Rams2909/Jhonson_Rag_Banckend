"""
Cross-encoder reranker.

Falls back to RRF score ordering if the model is not yet downloaded,
so the pipeline never blocks on a 2GB download mid-request.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache

from app.config import settings

# Set cache dir so we can check if model exists before loading
_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")


def _model_cached() -> bool:
    """True if the reranker weights are already on disk."""
    model_slug = settings.reranker_model.replace("/", "--")
    model_dir = os.path.join(_CACHE_DIR, f"models--{model_slug}")
    return os.path.isdir(model_dir)


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings.reranker_model)


def rerank(query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
    k = top_k or settings.rerank_top_k
    start = time.perf_counter()

    if not chunks:
        return []

    # If model not downloaded yet, skip reranking — use RRF scores as-is
    if not _model_cached():
        print(f"[reranker] model not cached — skipping rerank, using RRF order (top {k})")
        result = []
        for chunk in chunks[:k]:
            result.append({**chunk, "rerank_score": chunk.get("rrf_score", 0.0)})
        return result

    reranker = _get_reranker()
    pairs = [(query, c["content"]) for c in chunks]
    scores = reranker.predict(pairs, show_progress_bar=False)

    scored = sorted(
        zip(chunks, scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    result = []
    for chunk, score in scored[:k]:
        result.append({**chunk, "rerank_score": round(float(score), 6)})

    elapsed = (time.perf_counter() - start) * 1000
    print(f"[LATENCY] agent=reranker duration={elapsed:.2f}ms top_k={k}")
    return result
