"""Optional HTTP interface around the same pipeline the CLI uses.

    uvicorn api:app --reload
    POST /screen   {"input_dir": "./resumes", "use_llm": true, "use_github": true}
    GET  /results  -> last run's results (JSON)
    GET  /         -> the HTML dashboard for the last run
    GET  /health
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from screener.config import load_config  # noqa: E402
from screener.pipeline import run_pipeline  # noqa: E402
from screener.report import write_outputs  # noqa: E402
from screener.report_html import render_html, write_html  # noqa: E402

DEFAULT_OUTPUT = ROOT / "output" / "results.json"
CONFIG_PATH = ROOT / "config.yaml"

app = FastAPI(title="Resume Screener", version="1.1.0")
_lock = threading.Lock()
_last: dict[str, Any] | None = None


class ScreenRequest(BaseModel):
    input_dir: str = "./resumes"
    output: str = str(DEFAULT_OUTPUT)
    use_llm: bool = True
    use_github: bool = True


def _current() -> dict[str, Any] | None:
    """The last run in memory, else the results file a previous run (CLI or API) left on disk."""
    if _last is not None:
        return _last
    try:
        return json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/screen")
async def screen(req: ScreenRequest) -> dict[str, Any]:
    global _last
    if not Path(req.input_dir).is_dir():
        raise HTTPException(status_code=400, detail=f"input_dir not found: {req.input_dir}")
    if not _lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A screening run is already in progress")
    try:
        results = await run_in_threadpool(run_pipeline, req.input_dir, load_config(CONFIG_PATH),
                                          req.use_llm, req.use_github)
        json_path, _ = write_outputs(results, req.output)
        write_html(results, json_path.with_suffix(".html"))
        _last = results
    finally:
        _lock.release()
    return {"summary": results["summary"], "top": [
        {k: c[k] for k in ("rank", "candidate_name", "total_score", "score_breakdown", "strengths")}
        for c in results["ranked_candidates"][:10]
    ]}


@app.get("/results")
def results() -> dict[str, Any]:
    data = _current()
    if data is None:
        raise HTTPException(status_code=404, detail="No screening run yet; POST /screen first")
    return data


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    data = _current()
    if data is None:
        return HTMLResponse("<h1>No results yet</h1><p>POST /screen first (see /docs).</p>", status_code=404)
    return HTMLResponse(render_html(data))
