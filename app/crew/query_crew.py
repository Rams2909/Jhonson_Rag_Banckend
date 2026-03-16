"""
GraphRAG query pipeline — 4 CrewAI agents with real-time progress events.

Progress events are pushed into an asyncio.Queue from the background thread
so the SSE stream reflects each agent finishing live.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator

from crewai import Agent, Crew, LLM, Process, Task

from app.config import settings
from app.retrieval.bm25_search import bm25_search
from app.retrieval.vector_search import vector_search
from app.retrieval.graph_search import graph_search
from app.retrieval.rrf_fusion import rrf_fuse
from app.reranker.cross_encoder import rerank
from app.utils.latency import LatencyTracker

os.environ.setdefault("OLLAMA_API_BASE", settings.ollama_base_url)

_executor = ThreadPoolExecutor(max_workers=4)

_SENTINEL = object()   # signals the queue is done


def _llm() -> LLM:
    return LLM(
        model=f"ollama/{settings.ollama_model}",
        base_url=settings.ollama_base_url,
        api_base=settings.ollama_base_url,
        timeout=120,
    )


def _parse_plan(raw: str) -> dict:
    for text in [raw, re.sub(r"```(?:json)?", "", raw).strip()]:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                continue
    return {"retrieval_modes": ["bm25", "vector", "graph"]}


def _run_pipeline(query: str, top_k: int, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> tuple[str, dict]:
    """Runs in a thread. Posts progress + final events into the queue."""

    def emit(event: str, data):
        asyncio.run_coroutine_threadsafe(queue.put({"event": event, "data": data}), loop)

    asyncio.set_event_loop(asyncio.new_event_loop())
    tracker = LatencyTracker()

    try:
        llm = _llm()

        # ── Agent 1: Query Planner ──────────────────────────────────────────
        emit("progress", {"agent": "query_planner", "status": "running"})
        with tracker.track("query_planner"):
            planner = Agent(
                role="Query Planner",
                goal="Analyse the query and decide which retrieval modes to use.",
                backstory="Expert at routing queries to the right retrieval strategies.",
                llm=llm, tools=[], verbose=True, allow_delegation=False,
            )
            plan_task = Task(
                description=(
                    f"Analyse this query: '{query}'\n\n"
                    "Return ONLY a valid JSON object, no extra text:\n"
                    '{"intent": "...", "retrieval_modes": ["bm25", "vector", "graph"]}\n\n'
                    "Use bm25 for keywords, vector for semantic meaning, graph for entities/relationships."
                ),
                expected_output='{"intent": "...", "retrieval_modes": [...]}',
                agent=planner,
            )
            crew1 = Crew(agents=[planner], tasks=[plan_task], process=Process.sequential, memory=False, verbose=True)
            plan_result = crew1.kickoff()

        plan = _parse_plan(str(plan_result))
        modes = plan.get("retrieval_modes", ["bm25", "vector", "graph"])
        emit("progress", {"agent": "query_planner", "status": "done", "ms": round(tracker._data.get("query_planner_ms", 0))})

        # ── Python: Retrieval ───────────────────────────────────────────────
        emit("progress", {"agent": "retriever", "status": "running"})
        with tracker.track("retriever"):
            bm25_hits   = bm25_search(query, settings.bm25_top_k)    if "bm25"   in modes else []
            vector_hits = vector_search(query, settings.vector_top_k) if "vector" in modes else []
            graph_hits  = graph_search(query, settings.graph_top_k)   if "graph"  in modes else []
        emit("progress", {"agent": "retriever", "status": "done", "ms": round(tracker._data.get("retriever_ms", 0)),
                           "hits": {"bm25": len(bm25_hits), "vector": len(vector_hits), "graph": len(graph_hits)}})

        # ── Python: Fusion + Rerank ─────────────────────────────────────────
        emit("progress", {"agent": "fusion_reranker", "status": "running"})
        with tracker.track("fusion_reranker"):
            fused  = rrf_fuse([bm25_hits, vector_hits, graph_hits], top_n=30)
            ranked = rerank(query, fused, top_k)
        emit("progress", {"agent": "fusion_reranker", "status": "done", "ms": round(tracker._data.get("fusion_reranker_ms", 0)),
                           "chunks": len(ranked)})

        # ── Agent 4: Answer Generator ────────────────────────────────────────
        emit("progress", {"agent": "answer_generator", "status": "running"})
        with tracker.track("answer_generator"):
            context = "\n\n".join(
                f"[{i+1}] (source={c.get('source','?')})\n{c['content']}"
                for i, c in enumerate(ranked)
            ) if ranked else "No relevant context found."

            answer_agent = Agent(
                role="Answer Generator",
                goal="Generate a precise answer using only the provided context.",
                backstory="Grounded assistant that only uses retrieved evidence to answer.",
                llm=llm, tools=[], verbose=True, allow_delegation=False,
            )
            answer_task = Task(
                description=(
                    f"Answer this question: {query}\n\n"
                    "Use ONLY the context below. Be concise and accurate.\n\n"
                    f"CONTEXT:\n{context}"
                ),
                expected_output="A clear, grounded answer.",
                agent=answer_agent,
            )
            crew2 = Crew(agents=[answer_agent], tasks=[answer_task], process=Process.sequential, memory=False, verbose=True)
            answer_result = crew2.kickoff()

        emit("progress", {"agent": "answer_generator", "status": "done", "ms": round(tracker._data.get("answer_generator_ms", 0))})
        return str(answer_result), tracker.totals()

    except Exception as exc:
        emit("error", str(exc))
        raise
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL), loop)


async def run_query_crew(query: str, top_k: int = 10) -> AsyncIterator[dict]:
    """
    Yields SSE events live as each agent finishes:
      {"event": "progress", "data": {"agent": "...", "status": "running"|"done", "ms": ...}}
      {"event": "token",    "data": "<word> "}
      {"event": "latency",  "data": {...}}
      {"event": "done",     "data": ""}
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    future = loop.run_in_executor(_executor, _run_pipeline, query, top_k, loop, queue)

    # Stream progress events as they arrive from the thread
    while True:
        item = await queue.get()
        if item is _SENTINEL:
            break
        yield item

    # Collect the final result
    try:
        answer, latency = await future
    except Exception as exc:
        yield {"event": "error", "data": str(exc)}
        yield {"event": "done",  "data": ""}
        return

    for word in answer.split(" "):
        yield {"event": "token", "data": word + " "}
        await asyncio.sleep(0)

    yield {"event": "latency", "data": latency}
    yield {"event": "done",    "data": ""}
