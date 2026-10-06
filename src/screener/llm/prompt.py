"""Prompt for resume analysis. The model extracts facts with quotes; it never assigns scores."""

from __future__ import annotations

import json

from ..extract import split_sections
from ..models import ResumeAnalysis

# Bump when the prompt or schema changes so cached responses are not reused.
PROMPT_VERSION = "2026-10-06.2"

# Sections that carry no project/skill evidence; dropped first when a resume is too long to send whole.
LOW_VALUE_SECTIONS = ("interests", "certifications", "education", "contact", "achievements")

SYSTEM_PROMPT = """You extract facts from a software engineering resume for a screening system.
You do NOT score or rank the candidate. You return JSON only.

Create one entry in "projects" for every project AND every internship/job (source = "work_experience").

For each entry, mark a signal present ONLY if the resume text explicitly shows it for THAT entry,
and put a short verbatim quote (max 20 words, copied exactly from the resume) in "evidence".
If you are not sure, set present=false and evidence="". Never invent or paraphrase evidence.

Signals:
- retrieval: RAG, embeddings, vector search, semantic search over documents/data.
- tool_calling: an LLM/agent calls tools, functions or external APIs as actions.
- orchestration: multi-step or multi-agent workflow, routing, planning, agent graphs (e.g. LangGraph, CrewAI, ADK).
- evaluation: output quality is measured (eval sets, metrics, accuracy, RAGAS, benchmarks, human review).
- state_memory: conversation memory, persisted agent state, checkpoints, session context.
- business_logic: meaningful data processing, backend/API/database work, or domain/product logic beyond calling a model.
- shipped: deployed, used by real users, or has measured impact (numbers).

ai_type: "llm_agentic" if the entry uses LLMs, RAG, embeddings or agents; "classical_ml" if only
traditional ML/deep learning (classifiers, CNNs, regression...); otherwise "none".

thin_wrapper = true when an llm_agentic entry is essentially "send a prompt to an LLM API and show the
answer", with no retrieval, tools, state, orchestration, evaluation or substantive data/product logic.

tutorial_style = true when the entry has no implementation details or evidence of ownership (only a
title and a tech list), or is a well-known tutorial clone with nothing added.

technologies: only technologies explicitly written for that entry. Do not infer from job titles."""


def build_user_prompt(resume_text: str, include_schema: bool) -> str:
    parts = [f"Resume text:\n<<<\n{resume_text}\n>>>"]
    if include_schema:
        schema = json.dumps(ResumeAnalysis.model_json_schema(), separators=(",", ":"))
        parts.append(f"Return a single JSON object that matches this JSON schema:\n{schema}")
    else:
        parts.append("Return the JSON object now.")
    return "\n\n".join(parts)


def compact_for_llm(text: str, max_chars: int) -> str:
    """Fit a resume into the prompt budget without silently losing projects or experience.

    A plain `text[:max_chars]` cuts the end of the resume, which is often a job or project.
    Instead, drop sections that carry no project evidence first, and only hard-truncate as a last resort.
    """
    if len(text) <= max_chars:
        return text
    sections = split_sections(text)
    kept = [body if name == "header" else f"{name.upper()}\n{body}"
            for name, body in sections.items() if name not in LOW_VALUE_SECTIONS]
    return "\n\n".join(kept)[:max_chars]
