"""CrewAI agent definitions for the GraphRAG query pipeline."""
from __future__ import annotations

import os

from crewai import Agent, LLM

from app.config import settings
from app.crew.tools import (
    BM25SearchTool,
    VectorSearchTool,
    GraphSearchTool,
    RRFFuseTool,
    RerankTool,
    ContextBuilderTool,
    LLMStreamTool,
)

# litellm reads OLLAMA_API_BASE from env to route requests correctly
os.environ.setdefault("OLLAMA_API_BASE", settings.ollama_base_url)


def _ollama_llm() -> LLM:
    return LLM(
        model=f"ollama/{settings.ollama_model}",
        base_url=settings.ollama_base_url,
        api_base=settings.ollama_base_url,
        timeout=120,
    )


def make_query_planner() -> Agent:
    return Agent(
        role="Query Planner",
        goal=(
            "Analyse the user query, detect intent, decompose into sub-queries if needed, "
            "and decide which retrieval modes are required."
        ),
        backstory=(
            "You are an expert at understanding natural language queries and mapping them "
            "to the most effective retrieval strategies: BM25 for keyword-heavy queries, "
            "vector search for semantic similarity, and graph search for relational lookups."
        ),
        llm=_ollama_llm(),
        tools=[],
        verbose=True,
        allow_delegation=False,
    )


def make_tri_modal_retriever() -> Agent:
    return Agent(
        role="Tri-Modal Retriever",
        goal=(
            "Execute retrieval across BM25, vector, and graph databases in parallel "
            "based on the Query Planner's decision."
        ),
        backstory=(
            "You are a retrieval specialist that queries multiple data sources simultaneously "
            "to gather the most relevant information from any angle."
        ),
        llm=_ollama_llm(),
        tools=[BM25SearchTool(), VectorSearchTool(), GraphSearchTool()],
        verbose=True,
        allow_delegation=False,
    )


def make_fusion_reranker() -> Agent:
    return Agent(
        role="Fusion & Reranker",
        goal=(
            "Apply Reciprocal Rank Fusion across the three result lists, then "
            "cross-encoder rerank to produce the top 10 most relevant chunks."
        ),
        backstory=(
            "You are a precision specialist. You merge and rank search results deterministically "
            "to ensure the highest quality context is surfaced for answer generation."
        ),
        llm=_ollama_llm(),
        tools=[RRFFuseTool(), RerankTool()],
        verbose=True,
        allow_delegation=False,
    )


def make_answer_generator() -> Agent:
    return Agent(
        role="Context Assembler & Answer Generator",
        goal=(
            "Build a clean, grounded prompt from the top-ranked chunks and graph context, "
            "then stream a precise answer via Ollama."
        ),
        backstory=(
            "You are the final stage of the pipeline. You synthesize retrieved evidence "
            "into a coherent, accurate answer while citing sources."
        ),
        llm=_ollama_llm(),
        tools=[ContextBuilderTool(), LLMStreamTool()],
        verbose=True,
        allow_delegation=False,
    )
