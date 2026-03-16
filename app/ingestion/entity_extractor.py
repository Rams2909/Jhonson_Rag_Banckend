"""Named entity extraction via Qwen 2.5 through Ollama."""
from __future__ import annotations

import json
import re

import httpx

from app.config import settings

_SYSTEM = (
    "You are a precise NER system. Extract named entities from the text. "
    "Return ONLY valid JSON with this schema: "
    '{"entities": [{"text": str, "type": str, "start": int}]}. '
    "Types: PERSON, ORG, PLACE, DATE, PRODUCT, CONCEPT, OTHER."
)


def _parse_entities(raw: str) -> list[dict]:
    """Robustly extract entities list even if Qwen wraps JSON in markdown."""
    # Try direct parse first
    try:
        return json.loads(raw).get("entities", [])
    except Exception:
        pass
    # Strip markdown code fences and retry
    clean = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        return json.loads(clean).get("entities", [])
    except Exception:
        pass
    # Find the first {...} block and parse that
    match = re.search(r'\{.*"entities".*\}', clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()).get("entities", [])
        except Exception:
            pass
    print(f"[entity_extractor] could not parse JSON from response: {raw[:200]}")
    return []


def extract_entities(text: str) -> list[dict]:
    """Return list of entity dicts from a chunk of text."""
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Extract entities from:\n\n{text}"},
        ],
        "stream": False,
        "format": "json",
    }
    try:
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
            timeout=60,          # Qwen can be slow on first run
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
        entities = _parse_entities(raw)
        print(f"[entity_extractor] extracted {len(entities)} entities from {len(text)} chars")
        return entities
    except Exception as exc:
        print(f"[entity_extractor] warning: {exc}")
        return []


def build_relations(entities: list[dict], doc_id: str) -> list[dict]:
    """
    Build co-occurrence relations between entities in the same chunk.
    Returns list of {source, target, relation, doc_id}.
    """
    relations: list[dict] = []
    for i, e1 in enumerate(entities):
        for e2 in entities[i + 1 :]:
            relations.append(
                {
                    "source": e1["text"],
                    "source_type": e1.get("type", "OTHER"),
                    "target": e2["text"],
                    "target_type": e2.get("type", "OTHER"),
                    "relation": "CO_OCCURS",
                    "doc_id": doc_id,
                }
            )
    return relations
