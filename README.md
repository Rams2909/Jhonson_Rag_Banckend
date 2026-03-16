# GraphRAG Backend

Production-grade Retrieval-Augmented Generation system using a knowledge graph + vector database + full-text search, orchestrated by a 4-agent CrewAI pipeline.

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Graph DB | FalkorDB (Redis-compatible) |
| Vector DB | PostgreSQL 16 + pgvector |
| LLM | Ollama (llama3.2 or any local model) |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-2-v2 |
| Agent Framework | CrewAI (4 agents, sequential) |
| Package Manager | uv |

---

## Architecture

### Ingestion Pipeline (`POST /ingest`)
```
Text Input
  → Hierarchical Chunker   (parent: 10 sentences, child: 3 sentences)
  → BGE Embeddings         → pgvector (HNSW index)
  → Ollama NER             → FalkorDB (entity nodes + CO_OCCURS edges)
```

### Query Pipeline (`POST /query` — SSE streaming)
```
Query
  → Agent 1: Query Planner      (LLM decides retrieval modes)
  → Agent 2: Tri-Modal Retriever (BM25 + vector KNN + graph traversal)
  → Agent 3: Fusion & Reranker   (RRF fusion + cross-encoder rerank)
  → Agent 4: Answer Generator    (LLM generates grounded answer)
  → SSE stream: progress | token | latency | done
```

---

## Project Structure

```
graphrag/
├── pyproject.toml              # uv project dependencies
├── docker-compose.yml          # FalkorDB + PostgreSQL services
├── .env.example                # environment variable template
├── setup/
│   └── init.sql                # pgvector extension + tables + indexes
└── app/
    ├── main.py                 # FastAPI app + all route definitions
    ├── config.py               # pydantic-settings config
    ├── ingestion/
    │   ├── chunker.py          # hierarchical text chunker
    │   ├── embedder.py         # BGE sentence embeddings
    │   └── entity_extractor.py # Ollama NER + relation builder
    ├── retrieval/
    │   ├── bm25_search.py      # PostgreSQL full-text search
    │   ├── vector_search.py    # pgvector KNN search
    │   ├── graph_search.py     # FalkorDB Cypher traversal
    │   └── rrf_fusion.py       # Reciprocal Rank Fusion
    ├── reranker/
    │   └── cross_encoder.py    # cross-encoder reranker (graceful fallback)
    ├── crew/
    │   ├── agents.py           # 4 CrewAI agent definitions
    │   ├── tasks.py            # task definitions
    │   ├── tools.py            # tool wrappers
    │   └── query_crew.py       # pipeline orchestration + SSE progress events
    ├── api/
    │   ├── ingest.py           # ingest endpoint logic
    │   ├── query.py            # SSE streaming query endpoint
    │   └── storage.py          # storage inspection + debug endpoints
    └── utils/
        └── latency.py          # LatencyTracker context manager
```

---

## Local Setup

### Prerequisites
- macOS / Linux
- Docker Desktop
- Python 3.11+

### Step 1 — Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### Step 2 — Install Ollama and pull the model
```bash
# macOS
brew install ollama
ollama serve &
ollama pull llama3.2
```

### Step 3 — Start databases
```bash
docker compose up -d
docker compose ps   # wait until both show "healthy"
```

### Step 4 — Configure environment
```bash
cp .env.example .env
# Edit .env if your ports or model name differ
```

### Step 5 — Install dependencies
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Step 6 — Start the API server
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API: `http://localhost:8000`
Docs: `http://localhost:8000/docs`

---

## API Reference

### `POST /ingest`
Ingest a document into pgvector + FalkorDB.

**Request**
```json
{
  "doc_id": "doc1",
  "text": "Your document text here...",
  "metadata": { "source": "manual" }
}
```

**Response**
```json
{
  "doc_id": "doc1",
  "chunks": 12,
  "entities": 24,
  "latency_ms": 430
}
```

---

### `POST /query`
Query the knowledge base. Returns a Server-Sent Events stream.

**Request**
```json
{
  "query": "Who founded Johnson & Johnson?",
  "top_k": 10
}
```

**SSE Events**
```
event: progress
data: {"agent": "query_planner", "status": "running"}

event: progress
data: {"agent": "query_planner", "status": "done", "ms": 3500}

event: token
data: Johnson & Johnson was founded...

event: latency
data: {"query_planner_ms": 3500, "retriever_ms": 200, "fusion_reranker_ms": 50, "answer_generator_ms": 9000, "total_ms": 12750}

event: done
data:
```

---

### `GET /storage/docs`
List all ingested documents with chunk counts.

### `GET /storage/docs/{doc_id}/chunks`
Get chunks for a document. Query param: `?level=parent|child`

### `GET /storage/docs/{doc_id}/entities`
Get entities and relationships from FalkorDB.

### `DELETE /storage/docs/{doc_id}`
Delete a specific document.

### `DELETE /storage/clear`
Wipe all storage (pgvector + FalkorDB).

### `GET /storage/debug`
Raw counts from both databases — useful for debugging.

### `GET /health/services`
Live health check for Ollama, pgvector, and FalkorDB.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_DSN` | `postgresql+psycopg://graphrag:graphrag@localhost:5432/graphrag` | PostgreSQL connection |
| `FALKORDB_HOST` | `localhost` | FalkorDB host |
| `FALKORDB_PORT` | `6379` | FalkorDB port |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base |
| `OLLAMA_MODEL` | `llama3.2` | Model name (must be pulled) |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-2-v2` | Reranker model |
| `TOP_K_DEFAULT` | `10` | Default results per query |

---

## Running Tests
```bash
uv run pytest -v
```

## Stopping Services
```bash
docker compose down
pkill ollama
```
