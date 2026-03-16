"""pgvector KNN search."""
from __future__ import annotations

import time

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from app.config import settings
from app.ingestion.embedder import embed_query


def vector_search(query: str, top_k: int | None = None) -> list[dict]:
    k = top_k or settings.vector_top_k
    start = time.perf_counter()

    q_vec = embed_query(query)
    dsn = settings.postgres_dsn.replace("postgresql+psycopg://", "postgresql://")

    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        register_vector(conn)
        rows = conn.execute(
            """
            SELECT
                id,
                doc_id,
                chunk_index,
                content,
                metadata,
                1 - (embedding <=> %s::vector) AS score
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (q_vec, q_vec, k),
        ).fetchall()

    elapsed = (time.perf_counter() - start) * 1000
    print(f"[LATENCY] agent=vector_search duration={elapsed:.2f}ms hits={len(rows)}")
    return [dict(r) | {"source": "vector"} for r in rows]
