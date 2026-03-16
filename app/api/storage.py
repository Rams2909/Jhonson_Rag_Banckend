"""GET/DELETE /storage — inspect and clear pgvector + FalkorDB."""
from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from app.config import settings
from app.retrieval.graph_search import get_graph


def _dsn() -> str:
    return settings.postgres_dsn.replace("postgresql+psycopg://", "postgresql://")


def list_docs() -> list[dict]:
    """All unique doc_ids with chunk counts."""
    with psycopg.connect(_dsn(), row_factory=dict_row) as conn:
        rows = conn.execute(
            """
            SELECT doc_id,
                   COUNT(*) AS total_chunks,
                   SUM(CASE WHEN metadata->>'level' = 'parent' THEN 1 ELSE 0 END) AS parent_chunks,
                   SUM(CASE WHEN metadata->>'level' = 'child'  THEN 1 ELSE 0 END) AS child_chunks,
                   MIN(created_at) AS ingested_at
            FROM chunks
            GROUP BY doc_id
            ORDER BY ingested_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_chunks(doc_id: str, level: str | None = None, limit: int = 50) -> list[dict]:
    """Chunks for a doc, optionally filtered by level (parent/child)."""
    with psycopg.connect(_dsn(), row_factory=dict_row) as conn:
        if level:
            rows = conn.execute(
                """
                SELECT id, chunk_index, content, metadata, created_at
                FROM chunks
                WHERE doc_id = %s AND metadata->>'level' = %s
                ORDER BY chunk_index
                LIMIT %s
                """,
                (doc_id, level, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, chunk_index, content, metadata, created_at
                FROM chunks
                WHERE doc_id = %s
                ORDER BY chunk_index
                LIMIT %s
                """,
                (doc_id, limit),
            ).fetchall()
    return [dict(r) for r in rows]


def get_entities(doc_id: str) -> dict:
    """Entities and relationships from FalkorDB for a doc_id."""
    graph = get_graph()
    entities: list[dict] = []
    relations: list[dict] = []

    try:
        res = graph.query(
            "MATCH (e:Entity {doc_id: $doc_id}) RETURN e.name AS name, e.type AS type",
            {"doc_id": doc_id},
        )
        for r in res.result_set:
            entities.append({"name": r[0], "type": r[1]})
    except Exception as exc:
        print(f"[storage] entity query warning: {exc}")

    try:
        res = graph.query(
            """
            MATCH (a:Entity {doc_id: $doc_id})-[r]->(b:Entity {doc_id: $doc_id})
            RETURN a.name AS source, type(r) AS relation, b.name AS target
            LIMIT 100
            """,
            {"doc_id": doc_id},
        )
        for r in res.result_set:
            relations.append({"source": r[0], "relation": r[1], "target": r[2]})
    except Exception as exc:
        print(f"[storage] relation query warning: {exc}")

    return {"entities": entities, "relations": relations}


def debug_storage() -> dict:
    """Raw counts + samples from both DBs — for troubleshooting."""
    out: dict = {"pgvector": {}, "falkordb": {}}

    with psycopg.connect(_dsn(), row_factory=dict_row) as conn:
        out["pgvector"]["total_chunks"] = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks"
        ).fetchone()["n"]
        out["pgvector"]["by_doc"] = [
            dict(r) for r in conn.execute(
                "SELECT doc_id, COUNT(*) AS chunks, "
                "SUM(CASE WHEN metadata->>'level'='parent' THEN 1 ELSE 0 END) AS parents, "
                "SUM(CASE WHEN metadata->>'level'='child'  THEN 1 ELSE 0 END) AS children "
                "FROM chunks GROUP BY doc_id"
            ).fetchall()
        ]
        sample = conn.execute(
            "SELECT doc_id, chunk_index, LEFT(content,80) AS preview, metadata "
            "FROM chunks ORDER BY id LIMIT 5"
        ).fetchall()
        out["pgvector"]["sample_chunks"] = [dict(r) for r in sample]

    graph = get_graph()
    try:
        res = graph.query("MATCH (e:Entity) RETURN COUNT(e) AS n")
        out["falkordb"]["total_entities"] = res.result_set[0][0] if res.result_set else 0
    except Exception as exc:
        out["falkordb"]["total_entities"] = f"error: {exc}"

    try:
        res = graph.query("MATCH ()-[r]->() RETURN COUNT(r) AS n")
        out["falkordb"]["total_relations"] = res.result_set[0][0] if res.result_set else 0
    except Exception as exc:
        out["falkordb"]["total_relations"] = f"error: {exc}"

    try:
        res = graph.query("MATCH (e:Entity) RETURN e.name, e.type, e.doc_id LIMIT 10")
        out["falkordb"]["sample_entities"] = [
            {"name": r[0], "type": r[1], "doc_id": r[2]} for r in res.result_set
        ]
    except Exception as exc:
        out["falkordb"]["sample_entities"] = f"error: {exc}"

    return out


def delete_doc(doc_id: str) -> dict:
    """Delete a single doc from pgvector + FalkorDB."""
    with psycopg.connect(_dsn()) as conn:
        deleted = conn.execute(
            "DELETE FROM chunks WHERE doc_id = %s", (doc_id,)
        ).rowcount
        conn.commit()

    graph = get_graph()
    try:
        graph.query(
            "MATCH (e:Entity {doc_id: $doc_id}) DETACH DELETE e",
            {"doc_id": doc_id},
        )
    except Exception as exc:
        print(f"[storage] graph delete warning: {exc}")

    return {"deleted_doc": doc_id, "chunks_removed": deleted}


def clear_all_storage() -> dict:
    """Wipe everything from pgvector + FalkorDB."""
    with psycopg.connect(_dsn()) as conn:
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.execute("TRUNCATE TABLE chunks RESTART IDENTITY")
        conn.commit()

    graph = get_graph()
    try:
        graph.query("MATCH (e:Entity) DETACH DELETE e")
    except Exception as exc:
        print(f"[storage] graph clear warning: {exc}")

    return {"cleared": True, "chunks_removed": total}
