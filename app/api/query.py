"""POST /query endpoint with SSE streaming."""
from __future__ import annotations

import asyncio
import json

from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request

from app.crew.query_crew import run_query_crew


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


async def stream_query(request: Request, body: QueryRequest):
    """Returns an SSE stream with token / latency / done events."""

    async def event_generator():
        async for event in run_query_crew(body.query, body.top_k):
            if await request.is_disconnected():
                break
            yield {
                "event": event["event"],
                "data": (
                    json.dumps(event["data"])
                    if isinstance(event["data"], dict)
                    else event["data"]
                ),
            }

    return EventSourceResponse(event_generator())
