"""Run the LLM (or the rules fallback) and verify its evidence against the resume text."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from .llm.base import LLMClient
from .llm.rules_fallback import RulesAnalyzer
from .models import SIGNAL_NAMES, ResumeAnalysis, Signal

log = logging.getLogger(__name__)


def analyze_resume(text: str, llm: LLMClient | None, fallback: RulesAnalyzer) -> tuple[ResumeAnalysis, str, list[str]]:
    """Returns (analysis, mode, notes). Never raises: any LLM problem falls back to rules."""
    notes: list[str] = []
    if llm is not None:
        try:
            analysis = llm.analyze(text)
            notes.extend(verify_evidence(analysis, text))
            return analysis, "llm", notes
        except Exception as exc:  # one bad model call must not affect the batch
            log.warning("LLM analysis failed, using rules fallback: %s", exc)
            notes.append(f"LLM analysis failed ({str(exc)[:160]}); scored with rules fallback")
    return fallback.analyze(text), "rules_fallback", notes


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _appears_in(quote: str, norm_text: str, threshold: float = 0.8) -> bool:
    q = _norm(quote)
    if not q:
        return False
    if q in norm_text:
        return True
    match = SequenceMatcher(None, norm_text, q, autojunk=False).find_longest_match(0, len(norm_text), 0, len(q))
    return match.size >= threshold * len(q)


def verify_evidence(analysis: ResumeAnalysis, text: str) -> list[str]:
    """Drop any LLM claim whose quote is not actually in the resume (anti-hallucination).

    Mutates `analysis` in place and returns human-readable notes about what was dropped.
    """
    norm_text = _norm(text)
    notes: list[str] = []
    kept = []
    for project in analysis.projects:
        verified_any = False
        for name in SIGNAL_NAMES:
            signal: Signal = getattr(project, name)
            if not signal.present:
                continue
            if _appears_in(signal.evidence, norm_text):
                verified_any = True
            else:
                setattr(project, name, Signal(present=False, evidence=""))
                notes.append(f"Dropped unverified '{name}' claim for '{project.title}'")
        project.technologies = [t for t in project.technologies if _norm(t) and _norm(t) in norm_text]
        if not verified_any and not _appears_in(project.title, norm_text, threshold=0.6):
            notes.append(f"Dropped project '{project.title}': not found in resume text")
            continue
        kept.append(project)
    analysis.projects = kept
    if analysis.candidate_name and not _appears_in(analysis.candidate_name, norm_text, threshold=0.9):
        notes.append(f"Ignored LLM name '{analysis.candidate_name}': not found in resume text")
        analysis.candidate_name = None
    return notes
