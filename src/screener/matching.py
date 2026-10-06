"""Word-bounded term matching with evidence lines.

Keyword matching is the backbone of the hard filter, so it has to avoid the classic
false positives: "ai" inside "maintain", "rag" inside "drag", "LL.M." read as "LLM".
"""

from __future__ import annotations

import re
from functools import lru_cache

_BOUNDARY_LEFT = r"(?<![A-Za-z0-9])"
_BOUNDARY_RIGHT = r"(?![A-Za-z0-9])"


@lru_cache(maxsize=4096)
def compile_term(term: str) -> re.Pattern[str]:
    """Compile a config term into a word-bounded regex (rules documented in config.yaml)."""
    if term.startswith("re:"):
        return re.compile(_BOUNDARY_LEFT + "(?:" + term[3:] + ")" + _BOUNDARY_RIGHT, re.IGNORECASE)

    # Flexible separators: "tool calling" == "tool-calling"; "next.js" == "nextjs".
    seps = {" ": r"[\s\-_]*", "-": r"[\s\-_]?", ".": r"\.?"}
    body = "".join(seps.get(ch, re.escape(ch)) for ch in term)
    if term[-1].isalpha():
        body += "s?"  # optional plural
    flags = 0 if (term.isupper() and len(term) <= 5) else re.IGNORECASE
    return re.compile(_BOUNDARY_LEFT + body + _BOUNDARY_RIGHT, flags)


def contains(text: str, terms: list[str]) -> bool:
    return any(compile_term(t).search(text) for t in terms)


def matched_terms(text: str, terms: list[str]) -> list[str]:
    """Return the config terms that appear in text (config spelling, deduplicated)."""
    return [t for t in terms if compile_term(t).search(text)]


def evidence_lines(text: str, terms: list[str], limit: int = 3) -> list[str]:
    """Return up to `limit` distinct lines of text that contain any of the terms."""
    found: list[str] = []
    patterns = [compile_term(t) for t in terms]
    for line in text.splitlines():
        stripped = clean_line(line)
        if not stripped:
            continue
        if any(p.search(stripped) for p in patterns) and stripped not in found:
            found.append(stripped)
            if len(found) >= limit:
                break
    return found


_BULLETS = "•●▪◦‣∙·-–—*>»✓✔➢➤►■□○◆"


def clean_line(line: str, max_len: int = 220) -> str:
    line = " ".join(line.split()).strip().lstrip(_BULLETS).strip()
    return line if len(line) <= max_len else line[: max_len - 1] + "…"
