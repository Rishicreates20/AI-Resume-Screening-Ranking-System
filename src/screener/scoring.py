"""Turn extracted evidence into the 100-point score.

The LLM never produces numbers. It (or the rules fallback) reports *signals* with quotes;
this module converts signals and keyword evidence into points using config weights.
Same evidence in -> same score out, which keeps ranking predictable and testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .extract import applied_text
from .matching import contains, evidence_lines
from .models import (SIGNAL_NAMES, EligibilityResult, GitHubResult, ProjectAnalysis, ResumeAnalysis,
                     ScoreBreakdown)

DEPTH_SIGNALS = ("retrieval", "tool_calling", "orchestration", "state_memory", "evaluation")
_PRETTY = {"retrieval": "retrieval", "tool_calling": "tool calling", "orchestration": "orchestration",
           "evaluation": "evaluation", "state_memory": "state/memory", "business_logic": "business logic",
           "shipped": "shipped/measured impact"}


@dataclass
class ScoreResult:
    breakdown: ScoreBreakdown
    total: float
    evidence: dict[str, list[str]] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    project_summary: str = ""


def _r(x: float) -> float:
    return round(x, 1)


# ---------------------------------------------------------------------------
# Keyword-evidence items (Python/backend, cloud)
# ---------------------------------------------------------------------------


def _score_item(item: dict[str, Any], used: str, full: str, listed_mult: float,
                extra_terms: list[str] | None = None) -> tuple[float, str, str]:
    """Best of: full terms used > partial used > full listed-only > partial listed-only.

    Returns (points, how, evidence_line). how in {"used", "listed", ""}.
    """
    terms = list(item["terms"]) + (extra_terms or [])
    options: list[tuple[float, str, list[str], str]] = [(item["points"], "used", terms, used),
                                                        (item["points"] * listed_mult, "listed", terms, full)]
    if item.get("partial_terms"):
        options += [(item["partial_points"], "used", item["partial_terms"], used),
                    (item["partial_points"] * listed_mult, "listed", item["partial_terms"], full)]
    best = (0.0, "", "")
    for pts, how, tms, text in options:
        if pts > best[0] and contains(text, tms):
            line = evidence_lines(text, tms, limit=1)
            best = (pts, how, line[0] if line else "")
    return best


def _keyword_category(items: list[dict[str, Any]], used: str, full: str, listed_mult: float,
                      python_libs: list[str]) -> tuple[float, list[str], list[str], list[str], list[str]]:
    """Returns (points, evidence, used_names, listed_only_names, missing_names)."""
    total, evidence, used_names, listed_names, missing = 0.0, [], [], [], []
    for item in items:
        extra = python_libs if item.get("implied_by_eligibility_libs") else None
        pts, how, line = _score_item(item, used, full, listed_mult, extra)
        total += pts
        if how == "used":
            used_names.append(item["name"])
            evidence.append(f"{item['name']} +{_r(pts)} (used): \"{line}\"")
        elif how == "listed":
            listed_names.append(item["name"])
            evidence.append(f"{item['name']} +{_r(pts)} (skills list only, no project evidence)")
        else:
            missing.append(item["name"])
    return total, evidence, used_names, listed_names, missing


# ---------------------------------------------------------------------------
# AI project depth
# ---------------------------------------------------------------------------


def _project_points(p: ProjectAnalysis, weights: dict[str, float], ml_cap: float) -> float:
    pts = sum(weights.get(n, 0) for n in SIGNAL_NAMES if getattr(p, n).present)
    return min(pts, ml_cap) if p.ai_type == "classical_ml" else pts


def _is_thin(p: ProjectAnalysis) -> bool:
    # Guard against contradictory model output: a project with real depth signals is not "thin".
    return p.ai_type == "llm_agentic" and p.thin_wrapper and not any(getattr(p, n).present for n in DEPTH_SIGNALS)


def _signals_of(p: ProjectAnalysis) -> list[str]:
    return [_PRETTY[n] for n in SIGNAL_NAMES if getattr(p, n).present]


def score_candidate(sections: dict[str, str], full_text: str, analysis: ResumeAnalysis,
                    eligibility: EligibilityResult, github: GitHubResult | None,
                    cfg: dict[str, Any]) -> ScoreResult:
    scfg = cfg["scoring"]
    mult = scfg.get("listed_only_multiplier", 0.5)
    python_libs = cfg["eligibility"].get("python_implied_terms", [])
    evidence: dict[str, list[str]] = {}
    strengths: list[str] = []
    concerns: list[str] = []

    # Where skills are *used*: project/experience sections + verified technologies & quotes from analysis.
    used_text, sections_found = applied_text(sections)
    # Verified quotes only: model-listed technologies could be copied from the skills list.
    analysis_text = "\n".join(getattr(p, n).evidence for p in analysis.projects for n in SIGNAL_NAMES
                              if getattr(p, n).present)
    used = used_text + "\n" + analysis_text

    # 1. AI / agentic project depth (40)
    acfg = scfg["ai_project_depth"]
    ai_projects = [p for p in analysis.projects if p.ai_type != "none"]
    ranked = sorted(((_project_points(p, acfg["signals"], acfg["classical_ml_project_cap"]), p) for p in ai_projects),
                    key=lambda t: -t[0])
    ai_ev: list[str] = []
    if ranked:
        best_pts, best = ranked[0]
        ai_score = best_pts
        ai_ev.append(f"Best AI project '{best.title}' = {_r(best_pts)} pts ({', '.join(_signals_of(best)) or 'no depth signals'})")
        for n in SIGNAL_NAMES:
            sig = getattr(best, n)
            if sig.present:
                ai_ev.append(f"  {_PRETTY[n]} +{acfg['signals'][n]}: \"{sig.evidence}\"")
        if best.ai_type == "classical_ml":
            ai_ev.append(f"  capped at {acfg['classical_ml_project_cap']} (classical ML, not an LLM/agentic system)")
        if len(ranked) > 1 and ranked[1][0] > 0:
            bonus = min(acfg["second_project_bonus_max"], acfg["second_project_bonus_fraction"] * ranked[1][0])
            ai_score += bonus
            ai_ev.append(f"Second AI project '{ranked[1][1].title}' bonus +{_r(bonus)}")
        ai_score = min(ai_score, acfg["max"])
    elif eligibility.ai_level != "none":
        ai_score = acfg["skills_only_points"]
        ai_ev.append(f"AI terms appear but no AI project describes how they were used: +{ai_score}")
        concerns.append("AI frameworks are listed, but no project shows how they were used")
    else:
        ai_score = 0
    evidence["ai_project_depth"] = ai_ev

    # 2. Python & backend (30)
    pcfg = scfg["python_backend"]
    py_score, py_ev, py_used, py_listed, py_missing = _keyword_category(pcfg["items"], used, full_text, mult, python_libs)
    py_score = min(py_score, pcfg["max"])
    evidence["python_backend"] = py_ev

    # 3. Cloud / deployment / full stack (15)
    ccfg = scfg["cloud_fullstack"]
    cl_score, cl_ev, cl_used, cl_listed, _ = _keyword_category(ccfg["items"], used, full_text, mult, [])
    fullstack = next((p for p in analysis.projects
                      if contains(" ".join(p.technologies), ccfg["frontend_terms"])
                      and contains(" ".join(p.technologies), ccfg["backend_terms"])), None)
    if fullstack:
        cl_score += ccfg["fullstack_points"]
        cl_ev.append(f"Full stack +{ccfg['fullstack_points']}: '{fullstack.title}' combines frontend and backend")
    cl_score = min(cl_score, ccfg["max"])
    evidence["cloud_fullstack"] = cl_ev

    # 4. Engineering depth (5): only counted where skills are used, never from a skills list
    ecfg = scfg["engineering_depth"]
    eng_hits = []
    for category, terms in ecfg["items"].items():
        lines = evidence_lines(used_text + "\n" + analysis_text, terms, limit=1)
        if lines:
            eng_hits.append(category)
            evidence.setdefault("engineering_depth", []).append(f"{category} +1: \"{lines[0]}\"")
    eng_score = min(len(eng_hits), ecfg["max"])

    # 5. GitHub (10)
    gh_score = github.score if github else 0
    evidence["github"] = [github.summary] if github and github.summary else []

    # Penalties
    pen_cfg = scfg["penalties"]
    penalty, pen_ev = 0.0, []
    llm_projects = [p for p in ai_projects if p.ai_type == "llm_agentic"]
    thin = [p for p in llm_projects if _is_thin(p)]
    if llm_projects and len(thin) == len(llm_projects):
        penalty += pen_cfg["all_ai_projects_thin_wrapper"]
        pen_ev.append(f"-{pen_cfg['all_ai_projects_thin_wrapper']}: every LLM project is a thin API wrapper ({', '.join(p.title for p in thin)})")
    elif thin:
        penalty += pen_cfg["some_ai_projects_thin_wrapper"]
        pen_ev.append(f"-{pen_cfg['some_ai_projects_thin_wrapper']}: thin LLM wrapper project(s): {', '.join(p.title for p in thin)}")
    tutorials = [p for p in analysis.projects if p.tutorial_style and p.source == "project"]
    if tutorials:
        tut_pen = min(pen_cfg["tutorial_style_max"], pen_cfg["tutorial_style_per_project"] * len(tutorials))
        penalty += tut_pen
        pen_ev.append(f"-{_r(tut_pen)}: project(s) listed without implementation detail: {', '.join(p.title for p in tutorials)}")
    penalty = min(penalty, pen_cfg["max_total"])
    evidence["penalties"] = pen_ev

    breakdown = ScoreBreakdown(
        ai_project_depth=_r(ai_score), python_backend=_r(py_score), cloud_fullstack=_r(cl_score),
        github=_r(gh_score), engineering_depth=_r(eng_score), penalties=-_r(penalty) if penalty else 0.0,
    )
    total = max(0.0, min(100.0, ai_score + py_score + cl_score + gh_score + eng_score - penalty))
    # Brief: strong Python without a meaningful AI project must not rank near the top.
    meaningful_ai = any(p.ai_type == "llm_agentic" and not _is_thin(p)
                        and any(getattr(p, n).present for n in DEPTH_SIGNALS) for p in ai_projects)
    cap = scfg.get("no_meaningful_ai_total_cap")
    if cap is not None and not meaningful_ai and total > cap:
        evidence["penalties"].append(f"Total capped at {cap} (from {_r(total)}): no meaningful LLM/agentic project")
        concerns.append(f"No meaningful LLM/agentic project: total capped at {cap}")
        total = cap
    total = _r(total)

    # Strengths / concerns (plain English, derived from the same evidence)
    if ranked and ranked[0][1].ai_type == "llm_agentic":
        best = ranked[0][1]
        depth = [_PRETTY[n] for n in DEPTH_SIGNALS if getattr(best, n).present]
        if len(depth) >= 2:
            strengths.append(f"Substantial LLM/agentic project '{best.title}' ({', '.join(depth)})")
        elif depth:
            strengths.append(f"LLM project with {depth[0]}: '{best.title}'")
    ai_work = [p for p in ai_projects if p.source == "work_experience"]
    if ai_work:
        strengths.append(f"AI work/internship experience: {ai_work[0].title}")
    if py_score >= 18 and py_used:
        strengths.append(f"Python backend used in real work: {', '.join(py_used)}")
    if cl_score >= 9:
        strengths.append(f"Cloud/deployment evidence: {', '.join(cl_used) or 'see breakdown'}")
    if eng_score >= 3:
        strengths.append(f"Engineering depth: {', '.join(eng_hits)}")
    if github and github.score >= 7:
        strengths.append(f"Active GitHub ({github.summary})")
    if github and github.matched_resume_projects:
        strengths.append(f"Resume project(s) found on GitHub: {', '.join(github.matched_resume_projects[:3])}")

    if not ranked and eligibility.ai_level == "none":
        concerns.append("No AI project evidence")
    if ranked and all(p.ai_type == "classical_ml" for _, p in ranked):
        concerns.append("Only classical ML projects; no LLM/RAG/agentic system")
    if thin:
        concerns.append(f"Thin LLM wrapper project(s): {', '.join(p.title for p in thin)}")
    if tutorials:
        concerns.append(f"Projects without implementation detail: {', '.join(p.title for p in tutorials)}")
    if py_listed + cl_listed:
        concerns.append(f"Listed in skills but not shown in projects: {', '.join(py_listed + cl_listed)}")
    if py_missing:
        concerns.append(f"No evidence of: {', '.join(py_missing)}")
    if github is not None and github.status in ("not_found", "rate_limited", "error"):
        concerns.append(f"GitHub enrichment unavailable ({github.status}); GitHub points not awarded")
    elif github is not None and github.status == "no_profile":
        concerns.append("No GitHub profile on resume (not penalised, but unverified)")
    if not sections_found:
        concerns.append("Resume sections could not be detected; evidence matched on the whole text")

    # Project summary: best AI project, else the most substantial other project
    summary_project = ranked[0][1] if ranked else next(iter(analysis.projects), None)
    project_summary = ""
    if summary_project:
        sig = ", ".join(_signals_of(summary_project))
        project_summary = f"{summary_project.title}: {summary_project.summary}".strip()
        if sig:
            project_summary += f" [signals: {sig}]"

    return ScoreResult(breakdown=breakdown, total=total, evidence=evidence, strengths=strengths[:5],
                       concerns=concerns[:6], project_summary=project_summary[:400])


def match_projects_to_repos(projects: list[ProjectAnalysis], repo_names: list[str]) -> list[str]:
    """Which resume projects have a similarly named public repo? (ownership signal, not scored)"""
    def slug(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    repos = {slug(r): r for r in repo_names if slug(r)}
    matched = []
    for p in projects:
        title = slug(p.title)
        if len(title) < 4:
            continue
        for rslug, rname in repos.items():
            if len(rslug) >= 4 and (rslug in title or title in rslug or SequenceMatcher(None, title, rslug).ratio() >= 0.8):
                matched.append(f"{p.title} -> {rname}")
                break
    return matched
