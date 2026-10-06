"""The single interface the pipeline depends on. Swap providers by adding one class."""

from __future__ import annotations

from typing import Protocol

from ..models import ResumeAnalysis


class LLMError(Exception):
    """Any failure to get a valid ResumeAnalysis from a model."""


class LLMClient(Protocol):
    name: str

    def analyze(self, resume_text: str) -> ResumeAnalysis:
        """Return a validated ResumeAnalysis or raise LLMError."""
        ...
