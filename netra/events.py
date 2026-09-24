"""A tiny pub/sub bus plus per-stage timing, so the console log and the dashboard
show exactly which accelerator ran each step and how long it took."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

log = logging.getLogger("netra")

Listener = Callable[[str, dict[str, Any]], None]


class EventBus:
    def __init__(self) -> None:
        self._listeners: list[Listener] = []
        self._lock = threading.Lock()

    def subscribe(self, fn: Listener) -> None:
        with self._lock:
            self._listeners.append(fn)

    def emit(self, kind: str, **data: Any) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(kind, data)
            except Exception:  # a broken listener must never break the assistant
                log.exception("event listener failed for %s", kind)

    @contextmanager
    def stage(self, name: str, engine: str):
        """Time a pipeline stage: `with bus.stage("asr", "Whisper-base · NPU"): ...`"""
        t0 = time.perf_counter()
        self.emit("stage_start", name=name, engine=engine)
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1000
            log.info("%-8s %7.0f ms  %s", name, ms, engine)
            self.emit("stage", name=name, engine=engine, ms=round(ms))


bus = EventBus()
