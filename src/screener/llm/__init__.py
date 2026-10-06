"""LLM adapters. Provider-specific code stays behind the LLMClient protocol."""

from .base import LLMClient, LLMError

__all__ = ["LLMClient", "LLMError"]
