"""Data models.

Two groups:
  * ResumeAnalysis & friends: the structured-output schema the LLM must return
    (the rules fallback produces the same schema, so scoring never cares which ran).
  * CandidateResult & friends: what the pipeline writes to results.json.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# LLM structured-output schema
# ---------------------------------------------------------------------------

SIGNAL_NAMES = (
    "retrieval",
    "tool_calling",
    "orchestration",
    "evaluation",
    "state_memory",
    "business_logic",
    "shipped",
)


class Signal(BaseModel):
    present: bool = Field(description="True only if the resume text explicitly shows this.")
    evidence: str = Field(
        default="",
        description="Short verbatim quote (max ~20 words) copied from the resume that proves it. Empty if not present.",
    )


class ProjectAnalysis(BaseModel):
    title: str = Field(description="Project or role title as written in the resume.")
    source: Literal["project", "work_experience", "other"] = "project"
    summary: str = Field(default="", description="One factual sentence: what was built and how.")
    technologies: list[str] = Field(default_factory=list, description="Technologies explicitly named for this item.")
    ai_type: Literal["llm_agentic", "classical_ml", "none"] = Field(
        default="none",
        description="llm_agentic = uses LLMs/RAG/agents/embeddings; classical_ml = traditional ML/DL only; none = not AI.",
    )
    retrieval: Signal = Field(default_factory=lambda: Signal(present=False))
    tool_calling: Signal = Field(default_factory=lambda: Signal(present=False))
    orchestration: Signal = Field(default_factory=lambda: Signal(present=False))
    evaluation: Signal = Field(default_factory=lambda: Signal(present=False))
    state_memory: Signal = Field(default_factory=lambda: Signal(present=False))
    business_logic: Signal = Field(default_factory=lambda: Signal(present=False))
    shipped: Signal = Field(default_factory=lambda: Signal(present=False))
    thin_wrapper: bool = Field(
        default=False,
        description="True if this AI item is only a thin wrapper around an LLM API call with no real workflow, retrieval, state, data processing or product logic.",
    )
    tutorial_style: bool = Field(
        default=False,
        description="True if listed without implementation details or evidence of ownership (e.g. just a title and a tech list, or a well-known tutorial).",
    )
    quality_note: str = Field(default="", description="One short reason for the thin_wrapper / tutorial_style judgement.")


class ResumeAnalysis(BaseModel):
    candidate_name: str | None = Field(default=None, description="Candidate's full name, or null.")
    projects: list[ProjectAnalysis] = Field(
        default_factory=list, description="Every project and every internship/job, one entry each."
    )


# ---------------------------------------------------------------------------
# Pipeline output
# ---------------------------------------------------------------------------

Status = Literal["eligible", "rejected", "failed", "duplicate"]


class EligibilityResult(BaseModel):
    eligible: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    python_evidence: list[str] = Field(default_factory=list)
    ai_evidence: list[str] = Field(default_factory=list)
    ai_level: Literal["llm_agentic", "classical_ml", "none"] = "none"


class ScoreBreakdown(BaseModel):
    ai_project_depth: float = 0
    python_backend: float = 0
    cloud_fullstack: float = 0
    github: float = 0
    engineering_depth: float = 0
    penalties: float = 0  # stored as a negative number


class GitHubResult(BaseModel):
    status: Literal["ok", "no_profile", "not_found", "rate_limited", "error", "disabled", "skipped_ineligible"]
    username: str | None = None
    score: float = 0
    activity_points: float = 0
    repo_points: float = 0
    last_activity: str | None = None
    public_repos: int = 0
    relevant_repos: list[str] = Field(default_factory=list)
    repo_names: list[str] = Field(default_factory=list, exclude=True)  # used to match resume projects
    matched_resume_projects: list[str] = Field(default_factory=list)
    summary: str = ""
    error: str | None = None
    from_cache: bool = False


class CandidateResult(BaseModel):
    rank: int | None = None
    candidate_name: str
    file: str
    status: Status
    eligible: bool = False
    total_score: float | None = None
    score_breakdown: ScoreBreakdown | None = None
    score_evidence: dict[str, list[str]] = Field(default_factory=dict)
    email: str | None = None
    github_url: str | None = None
    matched_skills: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    eligibility_evidence: dict[str, list[str]] = Field(default_factory=dict)
    project_summary: str = ""
    github_summary: str = ""
    github: GitHubResult | None = None
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    analysis_mode: Literal["llm", "rules_fallback", "not_analyzed"] = "not_analyzed"
    analysis_notes: list[str] = Field(default_factory=list)
    duplicate_of: str | None = None
    error: str | None = None
