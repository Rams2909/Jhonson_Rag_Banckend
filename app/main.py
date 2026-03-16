from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.ingest import IngestRequest, IngestResponse, ingest_document
from app.api.query import QueryRequest, stream_query
from app.api.storage import list_docs, get_chunks, get_entities, delete_doc, clear_all_storage, debug_storage
from app.config import settings

app = FastAPI(
    title="GraphRAG API",
    description="Production GraphRAG with FalkorDB + pgvector + CrewAI",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(body: IngestRequest):
    return ingest_document(body)


@app.post("/query")
async def query(request: Request, body: QueryRequest):
    return await stream_query(request, body)


@app.get("/storage/docs")
def storage_docs():
    return list_docs()


@app.get("/storage/docs/{doc_id}/chunks")
def storage_chunks(doc_id: str, level: str | None = None, limit: int = 50):
    return get_chunks(doc_id, level, limit)


@app.get("/storage/docs/{doc_id}/entities")
def storage_entities(doc_id: str):
    return get_entities(doc_id)


@app.delete("/storage/docs/{doc_id}")
def storage_delete_doc(doc_id: str):
    return delete_doc(doc_id)


@app.delete("/storage/clear")
def storage_clear():
    return clear_all_storage()


@app.get("/storage/debug")
def storage_debug():
    return debug_storage()


@app.get("/health/services")
def health_services():
    """Check all backing services — Ollama, pgvector, FalkorDB."""
    import httpx
    import psycopg
    from app.retrieval.graph_search import get_graph

    result: dict = {}

    # Ollama
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        qwen_present = any(settings.ollama_model in m for m in models)
        result["ollama"] = {
            "status": "ok",
            "models": models,
            "qwen_ready": qwen_present,
        }
    except Exception as exc:
        result["ollama"] = {"status": "error", "detail": str(exc), "qwen_ready": False}

    # pgvector
    try:
        dsn = settings.postgres_dsn.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(dsn) as conn:
            count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        result["pgvector"] = {"status": "ok", "total_chunks": count}
    except Exception as exc:
        result["pgvector"] = {"status": "error", "detail": str(exc)}

    # FalkorDB
    try:
        graph = get_graph()
        res = graph.query("MATCH (e:Entity) RETURN COUNT(e) AS n")
        result["falkordb"] = {
            "status": "ok",
            "total_entities": res.result_set[0][0] if res.result_set else 0,
        }
    except Exception as exc:
        result["falkordb"] = {"status": "error", "detail": str(exc)}

    return result


