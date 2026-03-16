from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # PostgreSQL
    postgres_dsn: str = "postgresql+psycopg://graphrag:graphrag@localhost:5432/graphrag"

    # FalkorDB
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_graph: str = "graphrag"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5"

    # Embeddings
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Retrieval
    top_k_default: int = 10
    bm25_top_k: int = 20
    vector_top_k: int = 20
    graph_top_k: int = 20
    rerank_top_k: int = 10


settings = Settings()
