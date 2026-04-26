from __future__ import annotations

import contextlib
import threading
import time
from typing import Any, Callable, Iterator

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    class _TqdmFallback:
        def __init__(self, iterable=None, total=None, desc=None, initial=0, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.initial = initial
            self.n = initial

        def __iter__(self):
            iterable = self.iterable if self.iterable is not None else ()
            for item in iterable:
                yield item
                self.update(1)

        def update(self, n: int = 1) -> None:
            self.n += n

        def close(self) -> None:
            return None

    def _tqdm(iterable=None, *args, **kwargs):
        return _TqdmFallback(iterable=iterable, **kwargs)


_state = threading.local()


def _progress_sink() -> Callable[[dict[str, Any]], None] | None:
    return getattr(_state, "sink", None)


@contextlib.contextmanager
def progress_event_sink(
    sink: Callable[[dict[str, Any]], None] | None,
) -> Iterator[None]:
    previous = _progress_sink()
    _state.sink = sink
    try:
        yield
    finally:
        _state.sink = previous


class _ObservedTqdm:
    """Small tqdm proxy that emits structured progress events for the web demo."""

    def __init__(self, bar: Any, sink: Callable[[dict[str, Any]], None]) -> None:
        self._bar = bar
        self._sink = sink
        self._started_at = time.perf_counter()
        self._last_emit_at = 0.0
        self._last_n = None
        self._closed = False
        self._emit("started", force=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bar, name)

    def __iter__(self):
        for item in self._bar:
            yield item
            self._emit("running")
        self._emit("completed", force=True)

    def __enter__(self):
        if hasattr(self._bar, "__enter__"):
            self._bar.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        if hasattr(self._bar, "__exit__"):
            return self._bar.__exit__(exc_type, exc, tb)
        return None

    def update(self, n: int = 1) -> None:
        self._bar.update(n)
        self._emit("running")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._emit("completed", force=True)
        self._bar.close()

    def _emit(self, status: str, *, force: bool = False) -> None:
        now = time.perf_counter()
        current = int(getattr(self._bar, "n", 0) or 0)
        total_raw = getattr(self._bar, "total", None)
        total = int(total_raw) if total_raw is not None else None
        completed = bool(total is not None and current >= total)
        if completed:
            status = "completed"
        if (
            not force
            and self._last_n == current
            and now - self._last_emit_at < 0.25
        ):
            return
        if not force and now - self._last_emit_at < 0.2 and not completed:
            return
        self._last_emit_at = now
        self._last_n = current
        desc = str(getattr(self._bar, "desc", "") or "").strip()
        percent = (current / total) if total else None
        elapsed = max(0.0, now - self._started_at)
        rate = (current / elapsed) if elapsed and current else None
        if total:
            message = f"{desc}: {current}/{total}"
            if percent is not None:
                message += f" ({percent * 100:.1f}%)"
        else:
            message = f"{desc}: {current}"
        self._sink(
            {
                "type": "progress_update",
                "step": "runtime_progress",
                "message": message,
                "progress": {
                    "id": desc or f"progress-{id(self)}",
                    "label": desc or "Progress",
                    "current": current,
                    "total": total,
                    "percent": percent,
                    "elapsed_seconds": round(elapsed, 4),
                    "rate": rate,
                    "status": status,
                },
            }
        )


def tqdm(iterable=None, *args, **kwargs):
    bar = _tqdm(iterable, *args, **kwargs)
    sink = _progress_sink()
    if sink is None:
        return bar
    return _ObservedTqdm(bar, sink)


def emit_progress_event(event: dict[str, Any]) -> None:
    """Emit a structured runtime event when a web-demo sink is active."""
    sink = _progress_sink()
    if sink is None:
        return
    sink(dict(event))


__all__ = ["emit_progress_event", "progress_event_sink", "tqdm"]
