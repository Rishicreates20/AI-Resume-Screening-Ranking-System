import json
from datetime import datetime, timedelta, timezone

import pytest

from screener.analysis import analyze_resume, verify_evidence
from screener.github import GitHubClient, score_github
from screener.llm.base import LLMError
from screener.llm.openai_compatible import OpenAICompatibleClient
from screener.llm.rules_fallback import RulesAnalyzer
from screener.models import ProjectAnalysis, ResumeAnalysis, Signal

NOW = datetime(2026, 10, 6, tzinfo=timezone.utc)


# ---------------------------------------------------------------- LLM layer

class FailingLLM:
    name = "failing"

    def analyze(self, resume_text):
        raise LLMError("HTTP 500")


def test_llm_failure_falls_back_to_rules_for_that_resume(cfg, resume_text):
    analysis, mode, notes = analyze_resume(resume_text("strong_agentic"), FailingLLM(), RulesAnalyzer(cfg))
    assert mode == "rules_fallback"
    assert analysis.projects
    assert "LLM analysis failed" in notes[0]


def test_hallucinated_evidence_is_dropped():
    text = "Projects\nDoc Bot\n- Built a RAG chatbot over policy PDFs using FAISS"
    analysis = ResumeAnalysis(projects=[ProjectAnalysis(
        title="Doc Bot", ai_type="llm_agentic",
        retrieval=Signal(present=True, evidence="RAG chatbot over policy PDFs using FAISS"),
        evaluation=Signal(present=True, evidence="evaluated with RAGAS on 500 questions"),  # not in resume
    )])
    notes = verify_evidence(analysis, text)
    project = analysis.projects[0]
    assert project.retrieval.present
    assert not project.evaluation.present
    assert any("evaluation" in n for n in notes)


def test_invented_project_is_dropped():
    analysis = ResumeAnalysis(projects=[ProjectAnalysis(title="Quantum Trading Agent", ai_type="llm_agentic")])
    verify_evidence(analysis, "Projects\nTodo App\n- React todo list")
    assert analysis.projects == []


class FakeResponse:
    def __init__(self, status, body, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.text = json.dumps(body) if not isinstance(body, str) else body

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(json)
        return self.responses.pop(0)


def _llm_cfg(cfg):
    return {**cfg["llm"], "max_retries": 2, "max_retry_wait_seconds": 0}


def test_adapter_retries_on_429_then_parses_structured_output(cfg, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    good = {"choices": [{"message": {"content": json.dumps({"candidate_name": "Asha Rao", "projects": []})}}]}
    session = FakeSession([FakeResponse(429, {"error": "rate"}, {"retry-after": "1"}), FakeResponse(200, good)])
    client = OpenAICompatibleClient(_llm_cfg(cfg), "key", cache=None, session=session)
    assert client.analyze("resume").candidate_name == "Asha Rao"
    assert len(session.calls) == 2


def test_adapter_downgrades_json_schema_when_rejected(cfg, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    good = {"choices": [{"message": {"content": "```json\n{\"projects\": []}\n```"}}]}
    reject = FakeResponse(400, {"error": {"message": "response_format json_schema not supported"}})
    session = FakeSession([reject, FakeResponse(200, good)])
    client = OpenAICompatibleClient({**_llm_cfg(cfg), "extra_params": {}}, "key", cache=None, session=session)
    client.analyze("resume")
    assert session.calls[1]["response_format"] == {"type": "json_object"}


def test_adapter_raises_llm_error_after_invalid_json(cfg):
    bad = {"choices": [{"message": {"content": "not json at all"}}]}
    session = FakeSession([FakeResponse(200, bad)] * 3)
    client = OpenAICompatibleClient(_llm_cfg(cfg), "key", cache=None, session=session)
    with pytest.raises(LLMError):
        client.analyze("resume")


# ---------------------------------------------------------------- GitHub

def _repo(name, days_ago, language="Python", fork=False, description=""):
    return {"name": name, "fork": fork, "language": language, "description": description, "topics": [],
            "pushed_at": (NOW - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z"), "archived": False}


def test_github_scoring_rewards_recent_relevant_repos(cfg):
    repos = [_repo("rag-bot", 5), _repo("agent-kit", 40), _repo("site", 10, "JavaScript"),
             _repo("forked-llm", 1, fork=True), _repo("old-ml", 800)]
    result = score_github("asha", repos, None, cfg["github"], NOW)
    assert result.activity_points == 5          # pushed 5 days ago
    assert result.repo_points == 2              # rag-bot, agent-kit (fork and stale repo excluded)
    assert result.score == 7


def test_github_no_activity_scores_zero(cfg):
    result = score_github("ghost", [], None, cfg["github"], NOW)
    assert result.score == 0


class GHResponse(FakeResponse):
    pass


class GHSession:
    def __init__(self, responses):
        self.responses, self.headers, self.calls = list(responses), {}, 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return self.responses.pop(0)


def test_github_404_and_rate_limit_circuit_breaker(cfg):
    session = GHSession([
        GHResponse(404, {"message": "Not Found"}),
        GHResponse(403, {"message": "API rate limit exceeded"}, {"X-RateLimit-Remaining": "0"}),
    ])
    client = GitHubClient(cfg["github"], token=None, cache=None, session=session)
    assert client.enrich("nobody", NOW).status == "not_found"
    assert client.enrich("second", NOW).status == "rate_limited"
    assert client.enrich("third", NOW).status == "rate_limited"
    assert session.calls == 2  # breaker stopped the third call
    assert client.enrich(None, NOW).status == "no_profile"
