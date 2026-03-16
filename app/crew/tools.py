"""CrewAI tool wrappers for all retrieval + fusion + rerank + LLM operations."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.config import settings
from app.retrieval.bm25_search import bm25_search
from app.retrieval.vector_search import vector_search
from app.retrieval.graph_search import graph_search
from app.retrieval.rrf_fusion import rrf_fuse
from app.reranker.cross_encoder import rerank


# ── Input schemas ────────────────────────────────────────────────────────────

class SearchInput(BaseModel):
    query: str = Field(description="Search query string")
    top_k: int = Field(default=20, description="Number of results to return")


class FuseInput(BaseModel):
    bm25_results: list[dict] = Field(description="BM25 ranked hits")
    vector_results: list[dict] = Field(description="Vector ranked hits")
    graph_results: list[dict] = Field(description="Graph ranked hits")
    top_n: int = Field(default=30, description="Top N after fusion")


class RerankInput(BaseModel):
    query: str = Field(description="Original query")
    chunks: list[dict] = Field(description="Chunks to rerank")
    top_k: int = Field(default=10, description="Top K after reranking")


class ContextBuilderInput(BaseModel):
    query: str = Field(description="Original user query")
    chunks: list[dict] = Field(description="Top-K reranked chunks")


class LLMStreamInput(BaseModel):
    prompt: str = Field(description="Full prompt to send to LLM")


# ── Tools ─────────────────────────────────────────────────────────────────────

class BM25SearchTool(BaseTool):
    name: str = "bm25_search_tool"
    description: str = "Full-text BM25 search over ingested document chunks."
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str, top_k: int = 20) -> str:
        results = bm25_search(query, top_k)
        return json.dumps(results)


class VectorSearchTool(BaseTool):
    name: str = "vector_search_tool"
    description: str = "Semantic vector KNN search using pgvector embeddings."
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str, top_k: int = 20) -> str:
        results = vector_search(query, top_k)
        return json.dumps(results)


class GraphSearchTool(BaseTool):
    name: str = "graph_search_tool"
    description: str = "Graph traversal search over FalkorDB entity relationships."
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str, top_k: int = 20) -> str:
        results = graph_search(query, top_k)
        return json.dumps(results)


class RRFFuseTool(BaseTool):
    name: str = "rrf_fuse_tool"
    description: str = "Reciprocal Rank Fusion across BM25, vector, and graph result lists."
    args_schema: type[BaseModel] = FuseInput

    def _run(
        self,
        bm25_results: list[dict],
        vector_results: list[dict],
        graph_results: list[dict],
        top_n: int = 30,
    ) -> str:
        start = time.perf_counter()
        fused = rrf_fuse([bm25_results, vector_results, graph_results], top_n=top_n)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[LATENCY] agent=rrf_fusion duration={elapsed:.2f}ms merged={len(fused)}")
        return json.dumps(fused)


class RerankTool(BaseTool):
    name: str = "rerank_tool"
    description: str = "Cross-encoder reranking of fused chunks using bge-reranker-v2-m3."
    args_schema: type[BaseModel] = RerankInput

    def _run(self, query: str, chunks: list[dict], top_k: int = 10) -> str:
        ranked = rerank(query, chunks, top_k)
        return json.dumps(ranked)


class ContextBuilderTool(BaseTool):
    name: str = "context_builder_tool"
    description: str = "Assemble a clean RAG prompt from ranked chunks."
    args_schema: type[BaseModel] = ContextBuilderInput

    def _run(self, query: str, chunks: list[dict]) -> str:
        context_parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "?")
            score = chunk.get("rerank_score", chunk.get("rrf_score", 0))
            context_parts.append(
                f"[{i}] (source={source}, score={score:.4f})\n{chunk['content']}"
            )

        context = "\n\n".join(context_parts)
        prompt = (
            f"You are a helpful assistant. Answer the question using ONLY the context below.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER:"
        )
        return prompt


class LLMStreamTool(BaseTool):
    name: str = "llm_stream_tool"
    description: str = "Stream an answer from Ollama Qwen 2.5 and return the full text."
    args_schema: type[BaseModel] = LLMStreamInput

    def _run(self, prompt: str) -> str:
        """Returns the full streamed response as a single string."""
        start = time.perf_counter()
        full_response = ""

        with httpx.stream(
            "POST",
            f"{settings.ollama_base_url}/api/generate",
            json={"model": settings.ollama_model, "prompt": prompt, "stream": True},
            timeout=120,
        ) as resp:
            first_token = True
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        if first_token:
                            ttft = (time.perf_counter() - start) * 1000
                            print(f"[LATENCY] agent=llm_first_token duration={ttft:.2f}ms")
                            first_token = False
                        full_response += token
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

        elapsed = (time.perf_counter() - start) * 1000
        print(f"[LATENCY] agent=llm_total duration={elapsed:.2f}ms tokens={len(full_response.split())}")
        return full_response
