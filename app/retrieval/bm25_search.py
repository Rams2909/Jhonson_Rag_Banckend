"""BM25 full-text search using PostgreSQL's built-in tsvector + rank_bm25 in-memory fallback."""
from __future__ import annotations

import time

import psycopg
from psycopg.rows import dict_row

from app.config import settings


def bm25_search(query: str, top_k: int | None = None) -> list[dict]:
    k = top_k or settings.bm25_top_k
    start = time.perf_counter()

    dsn = settings.postgres_dsn.replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                doc_id,
                chunk_index,
                content,
                metadata,
                ts_rank_cd(
                    to_tsvector('english', content),
                    plainto_tsquery('english', %s)
                ) AS score
            FROM chunks
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, query, k),
        ).fetchall()

    elapsed = (time.perf_counter() - start) * 1000
    print(f"[LATENCY] agent=bm25_search duration={elapsed:.2f}ms hits={len(rows)}")
    return [dict(r) | {"source": "bm25"} for r in rows]
