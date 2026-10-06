"""Tiny thread-safe JSON file cache, so re-runs do not repeat LLM or GitHub calls."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any


class JsonFileCache:
    def __init__(self, directory: str | Path, ttl_hours: float | None = None):
        self.dir = Path(directory)
        self.ttl_seconds = ttl_hours * 3600 if ttl_hours else None
        self._lock = threading.Lock()

    @staticmethod
    def make_key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:32]

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        try:
            if self.ttl_seconds and time.time() - path.stat().st_mtime > self.ttl_seconds:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def put(self, key: str, value: Any) -> None:
        try:
            with self._lock:
                self.dir.mkdir(parents=True, exist_ok=True)
                tmp = self._path(key).with_suffix(".tmp")
                tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
                tmp.replace(self._path(key))
        except OSError:
            pass  # caching is an optimisation; never fail the run because of it
