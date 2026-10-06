from screener.eligibility import check_eligibility
from screener.extract import split_sections
from screener.llm.rules_fallback import RulesAnalyzer
from screener.models import GitHubResult, ProjectAnalysis, ResumeAnalysis, Signal
from screener.scoring import score_candidate

NO_GH = GitHubResult(status="no_profile")


def score_text(text, cfg, analysis=None, github=NO_GH):
    sections = split_sections(text)
    analysis = analysis or RulesAnalyzer(cfg).analyze(text)
    return score_candidate(sections, text, analysis, check_eligibility(sections, cfg), github, cfg)


def test_strong_agentic_candidate_scores_high(cfg, resume_text):
    result = score_text(resume_text("strong_agentic"), cfg)
    assert result.total >= 80
    assert result.breakdown.ai_project_depth >= 35
    assert result.evidence["ai_project_depth"], "score must come with evidence"


def test_strong_python_without_ai_project_ranks_below_real_ai_builders(cfg, resume_text):
    backend_only = score_text(resume_text("python_backend_no_ai_project"), cfg)
    rag_builder = score_text(resume_text("fullstack_js_python_ai"), cfg)
    assert backend_only.breakdown.python_backend == 30  # strong Python is still recognised...
    assert backend_only.total <= cfg["scoring"]["no_meaningful_ai_total_cap"]  # ...but capped
    assert rag_builder.total > backend_only.total


def test_skill_used_in_project_beats_skill_only_listed(cfg):
    listed = "A B\nSKILLS\nPython, FastAPI, LangChain\nPROJECTS\nBot\n- RAG chatbot with embeddings and evaluation metrics"
    used = "A B\nSKILLS\nPython\nPROJECTS\nBot\n- RAG chatbot built with FastAPI, embeddings and evaluation metrics"
    s_listed = score_text(listed, cfg).evidence["python_backend"]
    s_used = score_text(used, cfg).evidence["python_backend"]
    assert any("FastAPI +3" in e and "skills list only" in e for e in s_listed)
    assert any("FastAPI +6" in e and "(used)" in e for e in s_used)


def test_thin_wrapper_is_penalised(cfg, resume_text):
    result = score_text(resume_text("thin_wrapper"), cfg)
    assert result.breakdown.penalties <= -10
    assert any("thin" in c.lower() for c in result.concerns)


def _project(**signals):
    base = dict(title="Agent", ai_type="llm_agentic", technologies=[])
    base.update({k: Signal(present=True, evidence=v) for k, v in signals.items()})
    return ProjectAnalysis(**base)


def test_scoring_is_deterministic_given_llm_signals(cfg, resume_text):
    text = resume_text("strong_agentic")
    analysis = ResumeAnalysis(projects=[_project(retrieval="RAG over 12k support documents",
                                                 tool_calling="tool calling into Jira and Slack APIs")])
    first = score_text(text, cfg, analysis=analysis.model_copy(deep=True))
    second = score_text(text, cfg, analysis=analysis.model_copy(deep=True))
    assert first.breakdown == second.breakdown
    assert first.breakdown.ai_project_depth == 15  # retrieval 8 + tool calling 7


def test_contradictory_thin_wrapper_flag_is_ignored_when_depth_exists(cfg, resume_text):
    p = _project(retrieval="RAG over 12k support documents")
    p.thin_wrapper = True
    result = score_text(resume_text("strong_agentic"), cfg, analysis=ResumeAnalysis(projects=[p]))
    assert result.breakdown.penalties == 0


def test_github_score_is_added_and_capped(cfg, resume_text):
    gh = GitHubResult(status="ok", score=10, summary="active")
    result = score_text(resume_text("strong_agentic"), cfg, github=gh)
    assert result.breakdown.github == 10
    assert result.total <= 100
