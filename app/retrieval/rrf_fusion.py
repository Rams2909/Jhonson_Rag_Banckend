"""Reciprocal Rank Fusion across multiple ranked result lists."""
from __future__ import annotations

from collections import defaultdict


def rrf_fuse(
    ranked_lists: list[list[dict]],
    k: int = 60,
    top_n: int | None = None,
) -> list[dict]:
    """
    ranked_lists: each is an ordered list of result dicts with at least 'content' key.
    k: RRF constant (default 60 per the original paper).
    Returns fused list sorted by RRF score descending.
    """
    scores: dict[str, float] = defaultdict(float)
    items: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            key = item["content"]
            scores[key] += 1.0 / (k + rank)
            if key not in items:
                items[key] = item

    fused = sorted(items.values(), key=lambda x: scores[x["content"]], reverse=True)
    for item in fused:
        item["rrf_score"] = round(scores[item["content"]], 6)

    return fused[:top_n] if top_n else fused
