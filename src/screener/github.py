"""Lightweight public GitHub enrichment (max 10 points, never a hard requirement).

API budget: unauthenticated GitHub allows 60 requests/hour, so without a token we make
exactly ONE call per candidate (/users/{u}/repos, sorted by push date). With GITHUB_TOKEN
(5000/hour) we also read /users/{u}/events/public for activity outside own repos.

Failure policy: 404 -> not_found; rate limit -> trip a circuit breaker so the remaining
candidates are marked rate_limited without more calls; anything else -> error. Nothing
here raises into the pipeline.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

import requests

from .cache import JsonFileCache
from .matching import contains
from .models import GitHubResult

log = logging.getLogger(__name__)


class RateLimited(Exception):
    pass


class GitHubClient:
    def __init__(self, cfg: dict[str, Any], token: str | None, cache: JsonFileCache | None = None,
                 session: requests.Session | None = None):
        self.cfg = cfg
        self.api = cfg.get("api_url", "https://api.github.com").rstrip("/")
        self.timeout = cfg.get("timeout_seconds", 10)
        self.token = token
        self.cache = cache
        self.session = session or requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "resume-screener",
        })
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self._breaker = threading.Event()  # set once we are rate limited
        self._memo: dict[str, GitHubResult] = {}
        self._memo_lock = threading.Lock()

    # -- public -----------------------------------------------------------------

    def enrich(self, username: str | None, now: datetime | None = None) -> GitHubResult:
        if not username:
            return GitHubResult(status="no_profile", summary="No GitHub profile on resume")
        key = username.lower()
        with self._memo_lock:  # same profile on two resumes -> one lookup per run
            if key in self._memo:
                return self._memo[key].model_copy()
        result = self._enrich_uncached(username, now or datetime.now(timezone.utc))
        with self._memo_lock:
            self._memo[key] = result
        return result.model_copy()

    # -- internals --------------------------------------------------------------

    def _enrich_uncached(self, username: str, now: datetime) -> GitHubResult:
        base = GitHubResult(status="error", username=username)
        cache_key = JsonFileCache.make_key("gh-v1", username.lower(), "auth" if self.token else "anon")
        data = self.cache.get(cache_key) if self.cache else None
        from_cache = data is not None
        if data is None:
            if self._breaker.is_set():
                base.status, base.error = "rate_limited", "skipped: GitHub rate limit reached earlier in this run"
                base.summary = "GitHub not checked (rate limit reached)"
                return base
            try:
                data = self._fetch(username)
            except RateLimited as exc:
                self._breaker.set()
                log.warning("GitHub rate limit reached; skipping remaining lookups (set GITHUB_TOKEN to raise it)")
                base.status, base.error, base.summary = "rate_limited", str(exc), "GitHub not checked (rate limit reached)"
                return base
            except requests.RequestException as exc:
                base.error, base.summary = f"{type(exc).__name__}: {exc}"[:200], "GitHub lookup failed"
                return base
            if self.cache and data.get("status") in ("ok", "not_found"):
                self.cache.put(cache_key, data)

        if data.get("status") == "not_found":
            return GitHubResult(status="not_found", username=username, from_cache=from_cache,
                                summary=f"GitHub user '{username}' not found (or profile is private)")
        if data.get("status") != "ok":
            base.error = data.get("error")
            return base
        result = score_github(username, data["repos"], data.get("events"), self.cfg, now)
        result.from_cache = from_cache
        return result

    def _get(self, path: str, params: dict[str, Any] | None = None) -> requests.Response:
        resp = self.session.get(f"{self.api}{path}", params=params, timeout=self.timeout)
        remaining = resp.headers.get("X-RateLimit-Remaining")
        if resp.status_code in (403, 429) and (remaining == "0" or "rate limit" in resp.text.lower()):
            raise RateLimited(f"HTTP {resp.status_code}: rate limit exceeded")
        return resp

    def _fetch(self, username: str) -> dict[str, Any]:
        resp = self._get(f"/users/{username}/repos", {"per_page": 100, "sort": "pushed", "type": "owner"})
        if resp.status_code == 404:
            return {"status": "not_found"}
        if resp.status_code != 200:
            return {"status": "error", "error": f"HTTP {resp.status_code}: {resp.text[:150]}"}
        repos = [
            {k: r.get(k) for k in ("name", "fork", "language", "description", "topics", "pushed_at",
                                   "stargazers_count", "archived")}
            for r in resp.json()
        ]
        events = None
        if self.token:  # second call only when we have budget for it
            try:
                ev = self._get(f"/users/{username}/events/public", {"per_page": 100})
                if ev.status_code == 200:
                    events = [{"type": e.get("type"), "created_at": e.get("created_at")} for e in ev.json()]
            except requests.RequestException:
                events = None
        return {"status": "ok", "repos": repos, "events": events}


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def score_github(username: str, repos: list[dict[str, Any]], events: list[dict[str, Any]] | None,
                 cfg: dict[str, Any], now: datetime) -> GitHubResult:
    """Pure function: 0-5 for recent activity + 0-5 for maintained, relevant repositories."""
    own = [r for r in repos if not r.get("fork")]
    timestamps = [t for t in (_parse_ts(r.get("pushed_at")) for r in own) if t]
    timestamps += [t for t in (_parse_ts(e.get("created_at")) for e in (events or [])) if t]
    last = max(timestamps) if timestamps else None
    days_since = (now - last).days if last else None

    activity = 0.0
    if days_since is not None:
        for rule in cfg["activity_points"]:
            if days_since <= rule["within_days"]:
                activity = float(rule["points"])
                break

    window = cfg.get("maintained_within_days", 365)
    ai_terms = cfg.get("ai_repo_terms", [])
    relevant = []
    for r in own:
        pushed = _parse_ts(r.get("pushed_at"))
        if not pushed or (now - pushed).days > window or r.get("archived"):
            continue
        blob = " ".join([r.get("name") or "", r.get("description") or "", " ".join(r.get("topics") or [])])
        blob = blob.replace("-", " ").replace("_", " ")
        if (r.get("language") or "").lower() == "python" or contains(blob, ai_terms):
            relevant.append(r["name"])
    repo_points = float(min(len(relevant), cfg.get("max_repo_points", 5)))

    if days_since is None:
        recency = "no public activity found"
    else:
        recency = f"last public activity {days_since} day(s) ago"
    summary = (f"{recency}; {len(relevant)} maintained Python/AI repo(s) in the last {window} days "
               f"out of {len(own)} own public repos")
    return GitHubResult(
        status="ok", username=username, score=activity + repo_points, activity_points=activity,
        repo_points=repo_points, last_activity=last.date().isoformat() if last else None,
        public_repos=len(own), relevant_repos=relevant[:10],
        repo_names=[r["name"] for r in own if r.get("name")], summary=summary,
    )
