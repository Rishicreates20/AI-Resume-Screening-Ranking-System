import json
import re
import shutil
import subprocess

import pytest
from conftest import FIXTURES, ROOT, make_pdf

from screener.pipeline import run_pipeline
from screener.report_html import render_html, write_html


def _results(cfg, tmp_path):
    folder = tmp_path / "resumes"
    folder.mkdir()
    for name in ("strong_agentic", "fullstack_js_python_ai", "java_react_only"):
        make_pdf(folder / f"{name}.pdf", (FIXTURES / f"{name}.txt").read_text(encoding="utf-8"))
    return run_pipeline(folder, cfg, use_llm=False, use_github=False)


def _embedded_json(html):
    block = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(block.group(1))


def test_dashboard_embeds_results_and_untrusted_text_cannot_break_out_of_the_script_tag():
    evil = '</script><img src=x onerror=alert(1)><!-- & "quotes"'
    results = {"summary": {}, "ranked_candidates": [{"candidate_name": evil, "rank": 1}], "rejected_candidates": []}
    html = render_html(results)
    assert evil not in html                       # no raw '</script>' from resume text
    assert html.count("</script>") == 2           # the data block and the app script, nothing else
    assert _embedded_json(html) == results        # JSON.parse restores the original text exactly


def test_dashboard_is_written_for_a_real_batch(cfg, tmp_path):
    results = _results(cfg, tmp_path)
    path = write_html(results, tmp_path / "out" / "results.html")
    html = path.read_text(encoding="utf-8")
    assert "Asha Rao" in html and _embedded_json(html)["summary"]["eligible"] == results["summary"]["eligible"]
    assert "http://" not in html and "https://cdn" not in html   # self-contained: works offline


def test_summary_reports_stage_timings(cfg, tmp_path):
    stages = _results(cfg, tmp_path)["summary"]["stage_seconds"]
    assert list(stages) == ["ingest_extract", "eligibility", "analysis", "github", "scoring"]
    assert all(v >= 0 for v in stages.values())


def test_limit_processes_only_the_first_n_files(cfg, tmp_path):
    folder = tmp_path / "resumes"
    folder.mkdir()
    for name in ("strong_agentic", "fullstack_js_python_ai", "java_react_only"):
        make_pdf(folder / f"{name}.pdf", (FIXTURES / f"{name}.txt").read_text(encoding="utf-8"))
    results = run_pipeline(folder, cfg, use_llm=False, use_github=False, limit=2)
    assert results["summary"]["total_files"] == 2


def test_docx_resume_flows_through_the_whole_pipeline(cfg, tmp_path):
    import docx

    folder = tmp_path / "resumes"
    folder.mkdir()
    d = docx.Document()
    for line in (FIXTURES / "strong_agentic.txt").read_text(encoding="utf-8").splitlines():
        d.add_paragraph(line)
    d.save(folder / "asha.docx")
    results = run_pipeline(folder, cfg, use_llm=False, use_github=False)
    assert results["summary"]["failed_unreadable"] == 0
    top = results["ranked_candidates"][0]
    assert top["candidate_name"] == "Asha Rao" and top["file"] == "asha.docx" and top["total_score"] >= 70


def _app_script(html):
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)   # the data block has attributes, so it does not match
    assert len(scripts) == 1
    return scripts[0]


def test_dashboard_page_basics(cfg, tmp_path):
    """Structural guards: landmarks, live regions, reduced motion, and the [hidden] rule (an author
    display:flex silently overrides the hidden attribute without it, which once left a filter visible)."""
    html = render_html(_results(cfg, tmp_path))
    for needle in ('lang="en"', "<main", 'role="tablist"', 'aria-live="polite"', "prefers-reduced-motion",
                   'name="color-scheme"', "[hidden] { display: none !important; }", 'class="skip"', "<h1>"):
        assert needle in html, needle


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_dashboard_script_is_valid_javascript(cfg, tmp_path):
    js = tmp_path / "dashboard.js"
    js.write_text(_app_script(render_html(_results(cfg, tmp_path))), encoding="utf-8")
    result = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ----------------------------------------------------------------------------- API

fastapi_testclient = pytest.importorskip("fastapi.testclient")
pytest.importorskip("httpx")


@pytest.fixture
def client(tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(ROOT))
    import api

    monkeypatch.setattr(api, "DEFAULT_OUTPUT", tmp_path / "nothing-here" / "results.json")
    monkeypatch.setattr(api, "_last", None)
    return fastapi_testclient.TestClient(api.app)


def test_api_has_no_results_before_first_run(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/results").status_code == 404
    assert client.get("/").status_code == 404


def test_api_screen_then_results_then_dashboard(client, cfg, tmp_path):
    folder = tmp_path / "resumes"
    folder.mkdir()
    for name in ("strong_agentic", "java_react_only"):
        make_pdf(folder / f"{name}.pdf", (FIXTURES / f"{name}.txt").read_text(encoding="utf-8"))
    out = tmp_path / "out" / "results.json"

    resp = client.post("/screen", json={"input_dir": str(folder), "output": str(out), "use_llm": False, "use_github": False})
    assert resp.status_code == 200
    assert resp.json()["summary"]["eligible"] == 1 and resp.json()["top"][0]["candidate_name"] == "Asha Rao"
    assert out.exists() and out.with_suffix(".csv").exists() and out.with_suffix(".html").exists()

    assert client.get("/results").json()["summary"]["total_files"] == 2
    page = client.get("/")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html") and "Asha Rao" in page.text


def test_api_rejects_missing_input_dir(client, tmp_path):
    resp = client.post("/screen", json={"input_dir": str(tmp_path / "nope")})
    assert resp.status_code == 400


def test_csv_neutralises_spreadsheet_formulas_from_resume_text(tmp_path):
    from screener.report import write_outputs

    results = {
        "summary": {}, "failed_files": [], "duplicates": [],
        "ranked_candidates": [{"rank": 1, "candidate_name": "=HYPERLINK(\"http://evil\",\"x\")", "status": "eligible",
                               "total_score": 80, "score_breakdown": {"penalties": -5.0}, "email": None, "github_url": None,
                               "analysis_mode": "llm", "strengths": ["+cmd"], "concerns": [], "file": "a.pdf"}],
        "rejected_candidates": [{"candidate_name": "@SUM(1)", "rejection_reasons": ["No evidence of Python stack"], "file": "b.pdf"}],
    }
    _, csv_path = write_outputs(results, tmp_path / "r.json")
    rows = csv_path.read_text(encoding="utf-8-sig").splitlines()
    assert rows[1].startswith("1,\"'=HYPERLINK") or "'=HYPERLINK" in rows[1]
    assert "'@SUM(1)" in rows[2]
    assert ",-5.0," in rows[1]       # real numbers are left alone
