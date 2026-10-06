"""Rule-based extraction: sections, contact details, name, skills, project entries.

Resumes do not share a layout, so everything here is heuristic and defensive.
Anything that fails returns an empty value rather than raising.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .matching import clean_line, compile_term

# Canonical section -> heading variants (normalised: lowercase, '&' -> 'and', no punctuation).
SECTION_HEADINGS: dict[str, list[str]] = {
    "summary": ["summary", "professional summary", "profile", "profile summary", "objective",
                "career objective", "about me", "about", "professional profile"],
    "skills": ["skills", "technical skills", "key skills", "core skills", "skill set", "skillset",
               "technologies", "tech stack", "technical stack", "tools", "tools and technologies",
               "technologies and tools", "skills and tools", "core competencies", "competencies",
               "technical proficiency", "programming languages", "languages", "technical expertise",
               "skills and interests", "technical skills and interests", "relevant skills"],
    "experience": ["experience", "work experience", "professional experience", "internship",
                   "internships", "internship experience", "employment", "employment history",
                   "work history", "industry experience", "relevant experience", "experience and internships",
                   "work experience and internships", "professional experience and internships"],
    "projects": ["projects", "project", "personal projects", "academic projects", "key projects",
                 "notable projects", "project work", "selected projects", "technical projects",
                 "major projects", "projects and research", "relevant projects", "project experience",
                 "ai projects", "side projects", "open source", "open source contributions"],
    "education": ["education", "academic background", "academics", "academic details",
                  "educational qualifications", "educational qualification", "qualifications",
                  "education and training", "academic qualifications"],
    "certifications": ["certifications", "certification", "certificates", "courses", "coursework",
                       "relevant coursework", "training", "trainings", "licenses and certifications",
                       "certifications and courses", "online courses", "courses and certifications"],
    "achievements": ["achievements", "awards", "honors", "honours", "accomplishments", "hackathons",
                     "extracurricular", "extracurricular activities", "extra curricular activities",
                     "activities", "leadership", "positions of responsibility", "responsibilities",
                     "awards and achievements", "achievements and awards", "volunteering",
                     "co curricular activities", "competitions"],
    "publications": ["publications", "publication", "research", "research papers", "papers", "research experience",
                     "research and publications"],
    "interests": ["interests", "hobbies", "hobbies and interests", "personal interests"],
    "contact": ["contact", "contact information", "personal details", "personal information"],
}

_HEADING_LOOKUP = {variant: canon for canon, variants in SECTION_HEADINGS.items() for variant in variants}


def _normalise_heading(line: str) -> str:
    line = line.lower().replace("&", " and ")
    line = re.sub(r"[^a-z ]", " ", line)
    return " ".join(line.split())


def detect_heading(line: str) -> str | None:
    """Return the canonical section name if this line is a section heading."""
    raw = line.strip()
    if not raw or len(raw) > 45:
        return None
    norm = _normalise_heading(raw)
    if not norm or len(norm.split()) > 5:
        return None
    if norm in _HEADING_LOOKUP:
        return _HEADING_LOOKUP[norm]
    # Looser match only for lines that look like headings (ALL CAPS or ending with ':'),
    # so a project titled "Experience Tracker App" is not mistaken for a heading.
    if raw.isupper() and len(norm.split()) <= 4:
        for variant in sorted(_HEADING_LOOKUP, key=len, reverse=True):
            if re.search(rf"\b{re.escape(variant)}\b", norm):
                return _HEADING_LOOKUP[variant]
    return None


def split_sections(text: str) -> dict[str, str]:
    """Split resume text into canonical sections. Text before the first heading is 'header'."""
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    for line in text.splitlines():
        heading = detect_heading(line)
        if heading:
            current = heading
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items() if "\n".join(lines).strip()}


def text_outside(sections: dict[str, str], excluded: list[str]) -> str:
    return "\n".join(body for name, body in sections.items() if name not in excluded)


def applied_text(sections: dict[str, str]) -> tuple[str, bool]:
    """Text where skills are *used* (projects, experience, publications).

    Returns (text, sections_found). If the parser found no project/experience section
    at all, falls back to everything except the skills list so a layout we could not
    parse is not punished as "keyword-only".
    """
    used = [sections.get(k, "") for k in ("projects", "experience", "publications", "achievements")]
    used_text = "\n".join(t for t in used if t)
    if sections.get("projects") or sections.get("experience"):
        return used_text, True
    fallback = text_outside(sections, ["skills", "education", "certifications", "interests"])
    return fallback, False


# ---------------------------------------------------------------------------
# Contact details
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s\-().]{8,16}\d")
GITHUB_URL_RE = re.compile(r"github\.com/([A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38})(/[^\s)\]|,;]*)?", re.I)
GITHUB_IO_RE = re.compile(r"([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))\.github\.io", re.I)
# "GitHub: jodoe" -> jodoe. The lookahead rejects emails/URLs ("GitHub: jo@x.io"), which are not usernames.
GITHUB_LABEL_RE = re.compile(r"github\s*[:|\-–]\s*@?([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))\b(?![@.]\w)", re.I)
LINKEDIN_RE = re.compile(r"linkedin\.com/in/[A-Za-z0-9\-_%]+", re.I)

# github.com/<x> paths that are not user profiles
_GITHUB_RESERVED = {
    "features", "about", "login", "join", "topics", "orgs", "settings", "marketplace", "pricing",
    "collections", "sponsors", "apps", "site", "enterprise", "explore", "trending", "events",
    "issues", "pulls", "notifications", "search", "security", "readme", "github", "users",
    # neighbours in a typical header line: "GitHub | LinkedIn | Portfolio" must not yield a username
    "linkedin", "portfolio", "leetcode", "codeforces", "codechef", "kaggle", "gitlab", "twitter", "email",
    "website", "resume", "profile", "link", "hackerrank", "medium", "behance", "dribbble",
}


@dataclass
class Contacts:
    email: str | None = None
    phone: str | None = None
    github_username: str | None = None
    github_url: str | None = None
    linkedin: str | None = None
    github_candidates: list[str] = field(default_factory=list)


def extract_contacts(text: str, links: list[str]) -> Contacts:
    contacts = Contacts()
    mailtos = [l[7:].split("?")[0] for l in links if l.lower().startswith("mailto:")]
    emails = EMAIL_RE.findall(text) + mailtos  # visible text first; links can be stale
    if emails:
        contacts.email = emails[0].strip().lower()

    header = "\n".join(text.splitlines()[:15])
    phone = PHONE_RE.search(header) or PHONE_RE.search(text)
    if phone and 10 <= len(re.sub(r"\D", "", phone.group())) <= 13:
        contacts.phone = phone.group().strip()

    linked = LINKEDIN_RE.search(" ".join(links)) or LINKEDIN_RE.search(text)
    if linked:
        contacts.linkedin = "https://www." + linked.group().lower()

    contacts.github_username, contacts.github_candidates = _github_username(text, links)
    if contacts.github_username:
        contacts.github_url = f"https://github.com/{contacts.github_username}"
    return contacts


def _github_username(text: str, links: list[str]) -> tuple[str | None, list[str]]:
    """Pick the profile owner. A bare profile link beats repo links; links beat plain text."""
    profile_hits: list[str] = []
    repo_hits: list[str] = []
    # Undo line-wrapped URLs like "github.com/\nusername".
    joined_text = re.sub(r"github\.com/\s*\n\s*", "github.com/", text, flags=re.I)
    for source in (links, joined_text.split()):
        for chunk in source:
            for m in GITHUB_URL_RE.finditer(chunk):
                user, rest = m.group(1), (m.group(2) or "").strip("/")
                if user.lower() in _GITHUB_RESERVED:
                    continue
                (repo_hits if rest else profile_hits).append(user)
    candidates = profile_hits + repo_hits
    if not candidates:
        candidates = [m.group(1) for m in GITHUB_LABEL_RE.finditer(text) if m.group(1).lower() not in _GITHUB_RESERVED]
    if not candidates:
        candidates = [m.group(1) for m in GITHUB_IO_RE.finditer(" ".join(links) + " " + text)]
    if not candidates:
        return None, []
    # A profile link wins; then the most frequent owner (repo links reinforce it); then first seen.
    counts = Counter(c.lower() for c in candidates)
    profiles = {p.lower() for p in profile_hits}
    best_lower = max(counts, key=lambda u: (u in profiles, counts[u]))
    best = next(c for c in candidates if c.lower() == best_lower)
    return best, list(counts)


# ---------------------------------------------------------------------------
# Name
# ---------------------------------------------------------------------------

_NAME_STOPWORDS = {"resume", "curriculum", "vitae", "cv", "profile", "contact", "email", "phone",
                   "linkedin", "github", "portfolio", "address", "mobile", "india", "engineer",
                   "developer", "student", "intern", "b.tech", "btech", "software"}


def _looks_like_name(candidate: str) -> bool:
    words = candidate.replace(".", " ").split()
    return (1 <= len(words) <= 4 and all(re.fullmatch(r"[A-Z][A-Za-z'\-]*", w) for w in words)
            and not any(w.lower() in _NAME_STOPWORDS for w in words) and not detect_heading(candidate)
            and (len(words) >= 2 or len(words[0]) >= 3))


def guess_name(text: str, filename: str, hint: str | None = None) -> str:
    lines = [l.strip() for l in text.splitlines()]
    if hint and _looks_like_name(hint):
        # Names split over two lines ("Prathamesh" / "Patil"): join a following single word.
        if len(hint.split()) == 1 and hint in lines:
            nxt = lines[lines.index(hint) + 1] if lines.index(hint) + 1 < len(lines) else ""
            if _looks_like_name(nxt) and len(nxt.split()) == 1:
                hint = f"{hint} {nxt}"
        return hint.title() if hint.isupper() else hint
    for line in lines[:8]:
        if not line or detect_heading(line) or "@" in line or re.search(r"\d", line):
            continue
        cleaned = re.split(r"\s[|•·–-]\s", line)[0].strip()
        if len(cleaned.replace(".", " ").split()) >= 2 and _looks_like_name(cleaned):
            return cleaned.title() if cleaned.isupper() else cleaned
    stem = Path(filename).stem
    return re.sub(r"[_\-]+", " ", stem).strip().title() or filename


# ---------------------------------------------------------------------------
# Skills & project entries
# ---------------------------------------------------------------------------


def find_skills(text: str, taxonomy: dict[str, list[str]]) -> list[str]:
    return [name for name, aliases in taxonomy.items() if any(compile_term(a).search(text) for a in aliases)]


BULLET_RE = re.compile(r"^\s*[•●▪◦‣∙·\-–—*>»✓✔➢➤►■□○◆]\s*")
_DATE_RE = re.compile(r"(19|20)\d{2}|present|current|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", re.I)


_NOT_A_TITLE_RE = re.compile(
    r"^(?:role|github|link|live(?:\s+demo)?|demo|tech\s*stack|tools|duration|location|website)\s*(?:[:|\-–]|$)", re.I)
_WRAPPED_SENTENCE_ENDINGS = (",", " and", " or", " with", " using", " to", " for", " of", " in", " the", " a", " on", " by", " &")


def _is_entry_title(line: str) -> bool:
    """False for fragments that only look like titles: a link or 'Role: ...' line, a lone '|',
    or a sentence that was wrapped mid-clause. Such lines stay attached to the previous entry."""
    low = line.lower()
    if len(re.findall(r"[A-Za-z]", line)) < 3 or "http" in low or "www." in low:
        return False
    return not (_NOT_A_TITLE_RE.match(line) or low.endswith(_WRAPPED_SENTENCE_ENDINGS))


def split_entries(section_text: str) -> list[dict[str, str]]:
    """Split a projects/experience section into entries of {title, text}.

    Heuristic: bullet lines are details; a short non-bullet line that follows
    details starts a new entry. Long non-bullet lines are wrapped bullet text.
    """
    entries: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] | None = None
    seen_detail = False
    for raw in section_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        is_bullet = bool(BULLET_RE.match(line))
        looks_like_title = ((not is_bullet) and len(line) <= 90 and not line.endswith(".") and not line[0].islower()
                            and _is_entry_title(line))
        if current is None or (looks_like_title and seen_detail):
            current = {"title": [clean_line(line)], "lines": [line]}
            entries.append(current)
            seen_detail = False
            continue
        current["lines"].append(line)
        if is_bullet or len(line) > 90:
            seen_detail = True
    result = []
    for e in entries:
        title = e["title"][0]
        title = re.split(r"\s[|–—]\s|\s-\s", title)[0].strip() or title
        result.append({"title": title[:100], "text": "\n".join(e["lines"])})
    return result
