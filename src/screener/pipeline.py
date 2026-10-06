"""Orchestration: ingest -> extract -> hard filter -> analyse -> GitHub -> score -> rank.

run_pipeline() knows nothing about the CLI or HTTP, so both main.py and the optional
FastAPI app call the same function.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analysis import analyze_resume
from .cache import JsonFileCache
from .config import env_secret
from .eligibility import check_eligibility
from .extract import Contacts, extract_contacts, find_skills, guess_name, split_sections, text_outside
from .github import GitHubClient
from .ingest import discover_files, read_document, text_fingerprint
from .llm.base import LLMClient
from .llm.openai_compatible import OpenAICompatibleClient
from .llm.rules_fallback import RulesAnalyzer
from .models import CandidateResult, EligibilityResult, GitHubResult
from .scoring import match_projects_to_repos, score_candidate

log = logging.getLogger(__name__)


@dataclass
class _Parsed:
    rel: str
    text: str
    sections: dict[str, str]
    contacts: Contacts
    name: str
    skills: list[str]
    eligibility: EligibilityResult | None = None


def build_llm(cfg: dict[str, Any], use_llm: bool) -> tuple[LLMClient | None, str]:
    lcfg = cfg["llm"]
    if not use_llm or not lcfg.get("enabled", True):
        return None, "disabled (rules-only scoring)"
    key = env_secret(lcfg.get("api_key_env"))
    if not key:
        return None, f"{lcfg.get('api_key_env')} not set (rules-only scoring)"
    cache = JsonFileCache(lcfg.get("cache_dir", ".cache/llm"))
    return OpenAICompatibleClient(lcfg, key, cache), f"{lcfg['model']} via {lcfg['base_url']}"


def build_github(cfg: dict[str, Any], use_github: bool) -> GitHubClient | None:
    gcfg = cfg["github"]
    if not use_github or not gcfg.get("enabled", True):
        return None
    cache = JsonFileCache(gcfg.get("cache_dir", ".cache/github"), ttl_hours=gcfg.get("cache_ttl_hours", 24))
    return GitHubClient(gcfg, env_secret(gcfg.get("token_env")), cache)


def run_pipeline(input_dir: str | Path, cfg: dict[str, Any], use_llm: bool = True, use_github: bool = True,
                 llm: LLMClient | None = None, github: GitHubClient | None = None,
                 now: datetime | None = None, limit: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    lap = started

    def mark(stage: str) -> None:
        nonlocal lap
        now_ = time.perf_counter()
        stage_seconds[stage] = round(now_ - lap, 2)
        lap = now_

    input_dir = Path(input_dir)
    now = now or datetime.now(timezone.utc)

    if llm is None:
        llm, llm_label = build_llm(cfg, use_llm)
    else:
        llm_label = getattr(llm, "name", "custom client")
    if github is None:
        github = build_github(cfg, use_github)
    fallback = RulesAnalyzer(cfg)

    # 1. Ingest every file; failures are recorded, never raised.
    files = discover_files(input_dir)
    log.info("Found %d file(s) in %s", len(files), input_dir)
    if limit is not None and limit < len(files):
        log.info("--limit %d: processing only the first %d file(s)", limit, limit)
        files = files[:max(limit, 0)]
    failed: list[CandidateResult] = []
    duplicates: list[CandidateResult] = []
    parsed: list[_Parsed] = []
    by_hash: dict[str, str] = {}
    by_text: dict[str, str] = {}
    by_email: dict[str, str] = {}

    for path in files:
        rel = str(path.relative_to(input_dir))
        doc = read_document(path)
        if not doc.ok:
            log.warning("FAILED  %s: %s", rel, doc.error)
            failed.append(CandidateResult(candidate_name=path.stem, file=rel, status="failed", error=doc.error))
            continue

        # 2. Extract
        sections = split_sections(doc.text)
        contacts = extract_contacts(doc.text, doc.links)
        name = guess_name(doc.text, path.name, doc.name_hint)

        original = by_hash.get(doc.file_hash) or by_text.get(text_fingerprint(doc.text))
        reason = "identical file content"
        if not original and contacts.email and contacts.email in by_email:
            original, reason = by_email[contacts.email], f"same email ({contacts.email})"
        if original:
            log.info("DUPLICATE %s of %s (%s)", rel, original, reason)
            duplicates.append(CandidateResult(candidate_name=name, file=rel, status="duplicate",
                                              duplicate_of=original, email=contacts.email,
                                              error=f"Duplicate of {original}: {reason}"))
            continue
        by_hash[doc.file_hash] = rel
        by_text[text_fingerprint(doc.text)] = rel
        if contacts.email:
            by_email[contacts.email] = rel
        skill_text = text_outside(sections, cfg["eligibility"].get("excluded_sections", []))
        parsed.append(_Parsed(rel, doc.text, sections, contacts, name, find_skills(skill_text, cfg["skills_taxonomy"])))

    mark("ingest_extract")

    # 3. Hard eligibility filter (rules only)
    for p in parsed:
        p.eligibility = check_eligibility(p.sections, cfg)
    eligible = [p for p in parsed if p.eligibility.eligible]
    rejected = [p for p in parsed if not p.eligibility.eligible]
    log.info("Eligibility: %d eligible, %d rejected", len(eligible), len(rejected))
    mark("eligibility")

    # 4. Analyse eligible candidates only (bounded concurrency; LLM is the expensive step)
    workers = cfg["llm"].get("max_concurrency", 2) if llm else 8
    done = Counter()

    def _analyse(p: _Parsed):
        out = analyze_resume(p.text, llm, fallback)
        done["n"] += 1
        log.info("Analysed %d/%d: %s (%s)", done["n"], len(eligible), p.rel, out[1])
        return out

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        analyses = list(pool.map(_analyse, eligible))
    mark("analysis")

    # 5. GitHub enrichment for eligible candidates (bounded concurrency, cached, circuit breaker)
    if github is not None:
        with ThreadPoolExecutor(max_workers=max(1, cfg["github"].get("max_concurrency", 4))) as pool:
            gh_results = list(pool.map(lambda p: github.enrich(p.contacts.github_username, now), eligible))
    else:
        gh_results = [GitHubResult(status="disabled", username=p.contacts.github_username,
                                   summary="GitHub enrichment disabled") for p in eligible]

    mark("github")

    # 6. Score
    ranked: list[CandidateResult] = []
    for p, (analysis, mode, notes), gh in zip(eligible, analyses, gh_results):
        if gh.status == "ok":
            gh.matched_resume_projects = match_projects_to_repos(analysis.projects, gh.repo_names)
        score = score_candidate(p.sections, p.text, analysis, p.eligibility, gh, cfg)
        ranked.append(CandidateResult(
            candidate_name=analysis.candidate_name or p.name,
            file=p.rel, status="eligible", eligible=True,
            total_score=score.total, score_breakdown=score.breakdown, score_evidence=score.evidence,
            email=p.contacts.email, github_url=p.contacts.github_url, matched_skills=p.skills,
            eligibility_evidence={"python": p.eligibility.python_evidence, "ai": p.eligibility.ai_evidence},
            project_summary=score.project_summary, github_summary=gh.summary, github=gh,
            strengths=score.strengths, concerns=score.concerns, analysis_mode=mode, analysis_notes=notes,
        ))

    mark("scoring")

    # 7. Rank: total, then AI depth, then Python/backend, then name (deterministic)
    ranked.sort(key=lambda c: (-(c.total_score or 0), -c.score_breakdown.ai_project_depth,
                               -c.score_breakdown.python_backend, c.candidate_name.lower()))
    for i, c in enumerate(ranked, 1):
        c.rank = i

    rejected_results = [
        CandidateResult(candidate_name=p.name, file=p.rel, status="rejected", eligible=False,
                        email=p.contacts.email, github_url=p.contacts.github_url, matched_skills=p.skills,
                        rejection_reasons=p.eligibility.rejection_reasons,
                        eligibility_evidence={"python": p.eligibility.python_evidence, "ai": p.eligibility.ai_evidence})
        for p in rejected
    ]

    modes = Counter(c.analysis_mode for c in ranked)
    gh_status = Counter(c.github.status for c in ranked if c.github)
    summary = {
        "total_files": len(files),
        "parsed_successfully": len(parsed) + len(duplicates),
        "failed_unreadable": len(failed),
        "duplicates": len(duplicates),
        "unique_candidates": len(parsed),
        "eligible": len(eligible),
        "rejected": len(rejected),
        "analysis_mode": {"llm": modes.get("llm", 0), "rules_fallback": modes.get("rules_fallback", 0)},
        "github_enrichment": dict(gh_status),
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "stage_seconds": stage_seconds,
    }
    log.info("Done in %.1fs (%s)", summary["runtime_seconds"],
             ", ".join(f"{k} {v}s" for k, v in stage_seconds.items()))
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "run": {
            "input_dir": str(input_dir),
            "llm": llm_label,
            "llm_enabled": llm is not None,
            "github": ("disabled" if github is None else
                       "authenticated" if github.token else "unauthenticated (60 req/hour)"),
        },
        "summary": summary,
        "ranked_candidates": [c.model_dump(mode="json") for c in ranked],
        "rejected_candidates": [c.model_dump(mode="json", include={
            "candidate_name", "file", "status", "eligible", "email", "github_url", "matched_skills",
            "rejection_reasons", "eligibility_evidence"}) for c in rejected_results],
        "failed_files": [c.model_dump(mode="json", include={"candidate_name", "file", "status", "error"}) for c in failed],
        "duplicates": [c.model_dump(mode="json", include={"candidate_name", "file", "status", "duplicate_of", "error"})
                       for c in duplicates],
    }
