import time
from contextlib import contextmanager
from typing import Generator


class LatencyTracker:
    """Request-scoped latency accumulator."""

    def __init__(self) -> None:
        self._data: dict[str, float] = {}

    @contextmanager
    def track(self, name: str) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - start) * 1000
            self._data[f"{name}_ms"] = round(ms, 2)
            print(f"[LATENCY] agent={name} duration={ms:.2f}ms")

    def totals(self) -> dict[str, float]:
        result = dict(self._data)
        result["total_ms"] = round(sum(self._data.values()), 2)
        return result
