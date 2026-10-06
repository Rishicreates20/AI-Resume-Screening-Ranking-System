"""Deterministic keyword analyzer that fills the same ResumeAnalysis schema as the LLM.

Used when no API key is configured, when --no-llm is passed, or when the LLM fails
for a particular resume. Weaker at judging depth than the LLM, but it never fails
and every signal it reports is a line copied from the resume.
"""

from __future__ import annotations

from typing import Any

from ..extract import BULLET_RE, applied_text, find_skills, guess_name, split_entries, split_sections
from ..matching import contains, evidence_lines
from ..models import SIGNAL_NAMES, ProjectAnalysis, ResumeAnalysis, Signal


class RulesAnalyzer:
    name = "rules_fallback"

    def __init__(self, cfg: dict[str, Any]):
        self.signal_terms: dict[str, list[str]] = cfg["scoring"]["rules_fallback_signals"]
        self.tutorial_terms: list[str] = cfg["scoring"].get("tutorial_terms", [])
        self.tutorial_max_detail = cfg["scoring"].get("tutorial_max_detail_chars", 80)
        self.agentic_terms: list[str] = cfg["eligibility"]["ai_agentic_terms"]
        self.ml_terms: list[str] = cfg["eligibility"].get("classical_ml_terms", [])
        self.taxonomy: dict[str, list[str]] = cfg["skills_taxonomy"]

    def analyze(self, resume_text: str) -> ResumeAnalysis:
        sections = split_sections(resume_text)
        items: list[tuple[str, dict[str, str]]] = []
        for section, source in (("projects", "project"), ("experience", "work_experience"),
                                ("publications", "other")):
            if sections.get(section):
                items.extend((source, e) for e in split_entries(sections[section]))
        if not items:  # layout we could not parse: analyse the usable text as one block
            text, _ = applied_text(sections)
            if text:
                items.append(("other", {"title": "Unsectioned resume content", "text": text}))

        projects = [self._analyze_entry(source, entry) for source, entry in items]
        return ResumeAnalysis(candidate_name=guess_name(resume_text, ""), projects=projects)

    def _analyze_entry(self, source: str, entry: dict[str, str]) -> ProjectAnalysis:
        text = entry["text"]
        if contains(text, self.agentic_terms):
            ai_type = "llm_agentic"
        elif contains(text, self.ml_terms):
            ai_type = "classical_ml"
        else:
            ai_type = "none"

        title_line, _, detail_text = text.partition("\n")
        signals: dict[str, Signal] = {}
        for name in SIGNAL_NAMES:
            terms = self.signal_terms.get(name, [])
            # Prefer a descriptive bullet as evidence; fall back to the title/tech-stack line.
            lines = evidence_lines(detail_text, terms, limit=1) or evidence_lines(title_line, terms, limit=1)
            signals[name] = Signal(present=bool(lines), evidence=lines[0] if lines else "")

        depth = [signals[n].present for n in ("retrieval", "tool_calling", "orchestration", "state_memory", "evaluation")]
        thin = ai_type == "llm_agentic" and not any(depth) and not signals["business_logic"].present

        detail = text[len(text.splitlines()[0]):].strip() if text else ""
        tutorial = len(detail) < self.tutorial_max_detail or contains(text, self.tutorial_terms)

        detail_lines = [l.strip() for l in detail.splitlines() if l.strip()]
        # Experience entries open with company / dates / role lines: the summary is the first real
        # description (a bullet or a long sentence), not the date line.
        described = next((l for l in detail_lines if BULLET_RE.match(l)), None) or             next((l for l in detail_lines if len(l) >= 70), None)
        summary = (described or (detail_lines[0] if detail_lines else entry["title"])).lstrip("•●▪◦-–*> ").strip()[:200]

        reasons = []
        if thin:
            reasons.append("LLM used with no retrieval/tools/state/orchestration/evaluation or backend logic found")
        if tutorial:
            reasons.append("little or no implementation detail")
        return ProjectAnalysis(
            title=entry["title"],
            source=source,
            summary=summary,
            technologies=find_skills(text, self.taxonomy),
            ai_type=ai_type,
            thin_wrapper=thin,
            tutorial_style=tutorial,
            quality_note="; ".join(reasons),
            **signals,
        )
