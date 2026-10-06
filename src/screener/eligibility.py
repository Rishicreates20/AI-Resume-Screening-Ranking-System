"""Hard eligibility filter. Pure rules, deliberately kept outside the LLM.

A candidate is eligible only with BOTH:
  1. Python evidence: Python itself, or a Python-only library (FastAPI, Django, PyTorch...),
     outside education/certification/interest sections.
  2. AI evidence: LLM/RAG/agentic terms (or, configurably, classical ML) outside those sections.

The gate checks for presence of evidence and never for absence of other stacks:
Java/React/Next.js on a resume never cause a rejection by themselves.
"""

from __future__ import annotations

from typing import Any

from .extract import applied_text, text_outside
from .matching import evidence_lines, matched_terms
from .models import EligibilityResult


def check_eligibility(sections: dict[str, str], cfg: dict[str, Any]) -> EligibilityResult:
    ecfg = cfg["eligibility"]
    excluded = ecfg.get("excluded_sections", [])
    countable = text_outside(sections, excluded)
    excluded_text = "\n".join(sections.get(s, "") for s in excluded)

    python_terms = ecfg["python_terms"] + ecfg.get("python_implied_terms", [])
    python_hits = matched_terms(countable, python_terms)
    python_evidence = evidence_lines(countable, python_terms)

    agentic_hits = matched_terms(countable, ecfg["ai_agentic_terms"])
    # LLM/RAG/agent terms count anywhere outside the excluded sections (a named framework is evidence).
    # Classical ML is a much broader vocabulary ("Machine Learning Basics" in a skills line), so it must
    # be shown in a project or a job to count.
    ml_scope = applied_text(sections)[0] if ecfg.get("classical_ml_requires_applied_evidence", True) else countable
    ml_hits = matched_terms(ml_scope, ecfg.get("classical_ml_terms", []))
    if agentic_hits:
        ai_level, ai_terms, ai_scope = "llm_agentic", ecfg["ai_agentic_terms"], countable
    elif ml_hits:
        ai_level, ai_terms, ai_scope = "classical_ml", ecfg["classical_ml_terms"], ml_scope
    else:
        ai_level, ai_terms, ai_scope = "none", [], countable
    ai_evidence = evidence_lines(ai_scope, ai_terms) if ai_terms else []

    reasons: list[str] = []
    if not python_hits:
        if matched_terms(excluded_text, python_terms):
            reasons.append("No evidence of Python stack: Python appears only in education/certification sections, not as a skill or project technology")
        else:
            reasons.append("No evidence of Python stack")
    if ai_level == "none":
        if matched_terms(excluded_text, ecfg["ai_agentic_terms"] + ecfg.get("classical_ml_terms", [])):
            reasons.append("No AI/agentic project evidence: AI appears only in education/certification sections")
        elif matched_terms(countable, ecfg.get("classical_ml_terms", [])):
            reasons.append("No AI/agentic project evidence: machine learning is only listed as a skill, not shown in a project or job")
        else:
            reasons.append("No AI/agentic project evidence")
    elif ai_level == "classical_ml" and not ecfg.get("allow_classical_ml_only", True):
        reasons.append("Only classical ML evidence; no LLM/RAG/agentic project, framework or implementation")

    return EligibilityResult(
        eligible=not reasons,
        rejection_reasons=reasons,
        python_evidence=python_evidence,
        ai_evidence=ai_evidence,
        ai_level=ai_level,
    )
