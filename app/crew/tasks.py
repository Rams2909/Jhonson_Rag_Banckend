"""CrewAI task definitions — one task per agent."""
from __future__ import annotations

from crewai import Task

from app.crew.agents import (
    make_query_planner,
    make_tri_modal_retriever,
    make_fusion_reranker,
    make_answer_generator,
)


def make_tasks(query: str, top_k: int = 10):
    planner = make_query_planner()
    retriever = make_tri_modal_retriever()
    fusioner = make_fusion_reranker()
    generator = make_answer_generator()

    plan_task = Task(
        description=(
            f"Analyse this query: '{query}'\n\n"
            "Output a JSON object with:\n"
            "- intent: one-line description of what the user wants\n"
            "- sub_queries: list of sub-queries (or just the original if no decomposition needed)\n"
            "- retrieval_modes: list containing any of ['bm25', 'vector', 'graph']\n\n"
            "Rules:\n"
            "- Use 'bm25' for keyword/exact-match queries\n"
            "- Use 'vector' for semantic/conceptual queries\n"
            "- Use 'graph' for relationship/entity queries\n"
            "- Include all three for complex analytical queries\n\n"
            "Return ONLY valid JSON."
        ),
        expected_output=(
            '{"intent": "...", "sub_queries": ["..."], "retrieval_modes": ["bm25", "vector", "graph"]}'
        ),
        agent=planner,
    )

    retrieve_task = Task(
        description=(
            f"Using the query plan from the previous task, retrieve results for: '{query}'\n\n"
            "For each retrieval mode specified in the plan:\n"
            "- Run bm25_search_tool for BM25 mode\n"
            "- Run vector_search_tool for vector mode\n"
            "- Run graph_search_tool for graph mode\n\n"
            f"Use top_k={top_k * 2} for each search.\n\n"
            "Return a JSON object: {\"bm25\": [...], \"vector\": [...], \"graph\": [...]}"
        ),
        expected_output='{"bm25": [...], "vector": [...], "graph": [...]}',
        agent=retriever,
        context=[plan_task],
    )

    fuse_task = Task(
        description=(
            f"Take the retrieval results and produce the top {top_k} most relevant chunks.\n\n"
            "Steps:\n"
            "1. Call rrf_fuse_tool with the bm25, vector, and graph result lists\n"
            "2. Call rerank_tool on the fused results using the original query\n\n"
            f"Query: '{query}'\n"
            f"top_k: {top_k}\n\n"
            "Return a JSON array of the top reranked chunks with scores."
        ),
        expected_output="[{...chunk with rerank_score...}, ...]",
        agent=fusioner,
        context=[retrieve_task],
    )

    answer_task = Task(
        description=(
            f"Generate a grounded answer to: '{query}'\n\n"
            "Steps:\n"
            "1. Call context_builder_tool with the query and the top reranked chunks\n"
            "2. Call llm_stream_tool with the assembled prompt\n\n"
            "Return the full answer text."
        ),
        expected_output="A complete, grounded answer to the user's query.",
        agent=generator,
        context=[fuse_task],
    )

    return [plan_task, retrieve_task, fuse_task, answer_task], {
        "planner": planner,
        "retriever": retriever,
        "fusioner": fusioner,
        "generator": generator,
    }
