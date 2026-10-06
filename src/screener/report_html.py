"""Self-contained HTML dashboard rendered from the results dict.

One file, no server, no build step, no network: results.json is embedded and a small
vanilla-JS page renders the leaderboard, filters and a per-candidate evidence inspector.
Open it by double-clicking, or let the FastAPI app serve it at GET /.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TEMPLATE = Path(__file__).with_name("templates") / "dashboard.html"
_PLACEHOLDER = "__DATA_JSON__"
# "<", ">" and "&" become JSON unicode escapes (backslash-u + 4 hex digits), which are valid inside a JSON string.
_SCRIPT_UNSAFE = {ord(ch): "\\u%04x" % ord(ch) for ch in "<>&"}


def render_html(results: dict[str, Any]) -> str:
    """Return the dashboard page with `results` embedded as JSON.

    Resume text is untrusted, and the data lives inside a <script> tag: escaping < > & keeps a
    resume containing "</script>" from breaking out of it (the browser's JSON.parse restores them).
    """
    data = json.dumps(results, ensure_ascii=False).translate(_SCRIPT_UNSAFE)
    return TEMPLATE.read_text(encoding="utf-8").replace(_PLACEHOLDER, data, 1)


def write_html(results: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(results), encoding="utf-8")
    return path
