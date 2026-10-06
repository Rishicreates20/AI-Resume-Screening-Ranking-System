"""The LLM adapter against a real (local) HTTP server instead of mocked `requests` objects.

The fake server speaks the OpenAI-compatible /chat/completions protocol. Unless a failure is scripted,
it answers with a valid structured analysis, so the whole path is exercised: request payload,
auth header, JSON-schema response format, parsing, evidence verification, caching and fallbacks.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import FIXTURES, make_pdf

from screener.cache import JsonFileCache
from screener.llm.base import LLMError
from screener.llm.openai_compatible import OpenAICompatibleClient
from screener.llm.rules_fallback import RulesAnalyzer
from screener.pipeline import run_pipeline


@pytest.fixture
def server(cfg):
    state = {"script": [], "requests": []}
    rules = RulesAnalyzer(cfg)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep test output quiet
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            status = state["script"].pop(0) if state["script"] else 200
            if status == 200:
                prompt = body["messages"][1]["content"]
                resume = prompt.split("<<<\n", 1)[1].split("\n>>>", 1)[0]
                content = json.dumps(rules.analyze(resume).model_dump())
                payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            else:
                payload = json.dumps({"error": {"message": f"scripted {status}"}}).encode()
            self.send_response(status)
            if status == 429:
                self.send_header("retry-after", "0")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state["base_url"] = f"http://127.0.0.1:{httpd.server_port}/v1"
    yield state
    httpd.shutdown()
    httpd.server_close()


def _client(cfg, server, cache=None):
    llm_cfg = {**cfg["llm"], "base_url": server["base_url"], "max_retries": 3, "max_retry_wait_seconds": 0,
               "max_concurrency": 1}
    return OpenAICompatibleClient(llm_cfg, "test-key", cache=cache), llm_cfg


def _resumes(tmp_path, *names):
    folder = tmp_path / "resumes"
    folder.mkdir()
    for name in names:
        make_pdf(folder / f"{name}.pdf", (FIXTURES / f"{name}.txt").read_text(encoding="utf-8"))
    return folder


def test_full_pipeline_in_llm_mode_over_real_http(cfg, server, tmp_path):
    client, llm_cfg = _client(cfg, server)
    folder = _resumes(tmp_path, "strong_agentic", "fullstack_js_python_ai", "java_react_only")
    results = run_pipeline(folder, {**cfg, "llm": llm_cfg}, llm=client, use_github=False)

    assert results["summary"]["analysis_mode"] == {"llm": 2, "rules_fallback": 0}   # rejected resume never hits the LLM
    assert len(server["requests"]) == 2
    req = server["requests"][0]
    assert req["auth"] == "Bearer test-key" and req["path"].endswith("/chat/completions")
    assert req["body"]["response_format"]["type"] == "json_schema"
    assert "projects" in req["body"]["response_format"]["json_schema"]["schema"]["properties"]
    assert req["body"]["temperature"] == 0
    assert all(c["analysis_mode"] == "llm" for c in results["ranked_candidates"])


def test_rate_limit_and_server_errors_are_retried(cfg, server):
    server["script"] = [429, 500]
    client, _ = _client(cfg, server)
    analysis = client.analyze("Jo Doe\nPROJECTS\nBot\n- Built a RAG chatbot with FAISS embeddings and tool calling")
    assert analysis.projects
    assert len(server["requests"]) == 3        # 429 -> 500 -> 200


@pytest.mark.parametrize("status, hint", [(401, "GROQ_API_KEY"), (404, "llm.model")])
def test_bad_key_or_model_disables_llm_after_one_request(cfg, server, tmp_path, status, hint):
    """A configuration error will not fix itself: one clear message, no 40 identical failures."""
    server["script"] = [status] * 10
    client, llm_cfg = _client(cfg, server)
    folder = _resumes(tmp_path, "strong_agentic", "fullstack_js_python_ai", "classical_ml")
    results = run_pipeline(folder, {**cfg, "llm": llm_cfg}, llm=client, use_github=False)

    assert len(server["requests"]) == 1
    assert results["summary"]["analysis_mode"] == {"llm": 0, "rules_fallback": 3}   # batch still completes
    notes = results["ranked_candidates"][0]["analysis_notes"][0]
    assert f"HTTP {status}" in notes and hint in notes and "LLM disabled" in notes
    with pytest.raises(LLMError):
        client.analyze("another resume")
    assert len(server["requests"]) == 1


def test_response_cache_makes_reruns_free(cfg, server, tmp_path):
    client, _ = _client(cfg, server, cache=JsonFileCache(tmp_path / "cache"))
    text = "Jo Doe\nPROJECTS\nBot\n- Built a RAG chatbot with FAISS embeddings and tool calling"
    first = client.analyze(text)
    second = client.analyze(text)
    assert first == second and len(server["requests"]) == 1
