"""FalkorDB Cypher graph search."""
from __future__ import annotations

import time
from functools import lru_cache

from falkordb import FalkorDB

from app.config import settings


@lru_cache(maxsize=1)
def get_graph():
    db = FalkorDB(host=settings.falkordb_host, port=settings.falkordb_port)
    return db.select_graph(settings.falkordb_graph)


def graph_search(query: str, top_k: int | None = None) -> list[dict]:
    """
    Find entities matching the query via substring, then return
    the 2-hop neighbourhood as context chunks.
    """
    k = top_k or settings.graph_top_k
    start = time.perf_counter()

    graph = get_graph()
    # Extract key terms (simple whitespace split — Qwen handles intent)
    terms = [t.strip() for t in query.split() if len(t.strip()) > 3][:5]

    results: list[dict] = []
    seen: set[str] = set()

    for term in terms:
        try:
            res = graph.query(
                """
                MATCH (e:Entity)-[r]-(related:Entity)
                WHERE toLower(e.name) CONTAINS toLower($term)
                RETURN e.name AS entity, type(r) AS relation,
                       related.name AS related, e.doc_id AS doc_id
                LIMIT $k
                """,
                {"term": term, "k": k},
            )
            for record in res.result_set:
                key = f"{record[0]}|{record[2]}"
                if key not in seen:
                    seen.add(key)
                    results.append(
                        {
                            "content": f"{record[0]} --[{record[1]}]--> {record[2]}",
                            "doc_id": record[3],
                            "score": 1.0,
                            "source": "graph",
                            "metadata": {"entity": record[0], "relation": record[1]},
                        }
                    )
        except Exception as exc:
            print(f"[graph_search] warning: {exc}")

    elapsed = (time.perf_counter() - start) * 1000
    print(f"[LATENCY] agent=graph_search duration={elapsed:.2f}ms hits={len(results)}")
    return results[:k]
