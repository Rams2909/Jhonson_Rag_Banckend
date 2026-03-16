-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Document chunks table with vector embeddings
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    embedding   vector(384),           -- BAAI/bge-small-en-v1.5 dim
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast KNN search
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- BM25 full-text search index
CREATE INDEX IF NOT EXISTS chunks_content_fts_idx
    ON chunks USING gin(to_tsvector('english', content));

-- Doc-level index
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks(doc_id);
