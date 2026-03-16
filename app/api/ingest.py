"""POST /ingest endpoint."""
from __future__ import annotations

import time

import psycopg
from pgvector.psycopg import register_vector
from pydantic import BaseModel

from app.config import settings
from app.ingestion.chunker import hierarchical_chunk
from app.ingestion.embedder import embed_texts
from app.ingestion.entity_extractor import extract_entities, build_relations
from app.retrieval.graph_search import get_graph


class IngestRequest(BaseModel):
    text: str
    doc_id: str
    metadata: dict = {}


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int
    entities: int
    latency_ms: int


def ingest_document(req: IngestRequest) -> IngestResponse:
    start = time.perf_counter()

    # 1. Chunk
    chunks = hierarchical_chunk(req.text, req.doc_id, metadata=req.metadata)

    # 2. Embed
    texts = [c.content for c in chunks]
    embeddings = embed_texts(texts)

    # 3. Store vectors in pgvector
    dsn = settings.postgres_dsn.replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for chunk, emb in zip(chunks, embeddings):
                cur.execute(
                    """
                    INSERT INTO chunks (doc_id, chunk_index, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        chunk.doc_id,
                        chunk.chunk_index,
                        chunk.content,
                        psycopg.types.json.Jsonb(chunk.metadata),
                        emb,
                    ),
                )
        conn.commit()

    # 4. Extract entities from child chunks only
    all_entities: list[dict] = []
    graph = get_graph()

    for chunk in chunks:
        if chunk.metadata.get("level") != "child":
            continue
        entities = extract_entities(chunk.content)
        all_entities.extend(entities)
        relations = build_relations(entities, req.doc_id)

        for ent in entities:
            try:
                graph.query(
                    "MERGE (:Entity {name: $name, type: $type, doc_id: $doc_id})",
                    {"name": ent["text"], "type": ent.get("type", "OTHER"), "doc_id": req.doc_id},
                )
            except Exception as exc:
                print(f"[ingest] graph entity warning: {exc}")

        for rel in relations:
            try:
                graph.query(
                    """
                    MATCH (a:Entity {name: $source, doc_id: $doc_id})
                    MATCH (b:Entity {name: $target, doc_id: $doc_id})
                    MERGE (a)-[:CO_OCCURS]->(b)
                    """,
                    {
                        "source": rel["source"],
                        "target": rel["target"],
                        "doc_id": req.doc_id,
                    },
                )
            except Exception as exc:
                print(f"[ingest] graph relation warning: {exc}")

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return IngestResponse(
        doc_id=req.doc_id,
        chunks=len(chunks),
        entities=len(all_entities),
        latency_ms=elapsed_ms,
    )
