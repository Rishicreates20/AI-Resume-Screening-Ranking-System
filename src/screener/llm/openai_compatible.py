"""Adapter for any OpenAI-compatible /chat/completions API (Groq, OpenAI, OpenRouter, Gemini...).

Robustness built in:
  * structured output via JSON schema, validated again locally with Pydantic
  * retries with Retry-After / exponential backoff on 429, 5xx, timeouts and invalid JSON
  * automatic downgrade (json_schema -> json_object, drop unknown extra params) on HTTP 400
  * fail-fast on configuration errors (401/403/404: bad key or model name): the LLM is switched off for
    the rest of the run with ONE clear message instead of failing the same way for every resume
  * on-disk cache keyed by model + prompt version + resume text
Every failure ends in LLMError, which the pipeline turns into a rules fallback for that resume.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any

import requests
from pydantic import ValidationError

from ..cache import JsonFileCache
from ..models import ResumeAnalysis
from .base import LLMError
from .prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt, compact_for_llm

log = logging.getLogger(__name__)


class OpenAICompatibleClient:
    name = "openai_compatible"

    def __init__(self, cfg: dict[str, Any], api_key: str, cache: JsonFileCache | None = None,
                 session: requests.Session | None = None):
        self.url = cfg["base_url"].rstrip("/") + "/chat/completions"
        self.model = cfg["model"]
        self.api_key = api_key
        self.temperature = cfg.get("temperature", 0)
        self.max_tokens = cfg.get("max_output_tokens", 3000)
        self.timeout = cfg.get("timeout_seconds", 90)
        self.max_retries = cfg.get("max_retries", 4)
        self.max_wait = cfg.get("max_retry_wait_seconds", 60)
        self.max_chars = cfg.get("max_resume_chars", 12000)
        self.mode = cfg.get("structured_output_mode", "json_schema")
        self.extra_params = dict(cfg.get("extra_params") or {})
        self.cache = cache
        self.session = session or requests.Session()
        self.api_key_env = cfg.get("api_key_env", "the API key")
        self._lock = threading.Lock()
        self._fatal: str | None = None  # set once a config error makes further calls pointless

    # -- public ---------------------------------------------------------------

    def analyze(self, resume_text: str) -> ResumeAnalysis:
        text = compact_for_llm(resume_text, self.max_chars)
        key = JsonFileCache.make_key(self.model, PROMPT_VERSION, text)
        if self.cache and (hit := self.cache.get(key)) is not None:
            try:
                return ResumeAnalysis.model_validate(hit)
            except ValidationError:
                pass

        if self._fatal:
            raise LLMError(self._fatal)
        last_error: str = "no attempt made"
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(self.url, json=self._payload(text), headers=self._headers(),
                                         timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                self._sleep(attempt, None)
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                self._sleep(attempt, resp.headers.get("retry-after"))
                continue
            if resp.status_code == 400:
                body = resp.text
                if "json_validate_failed" in body:  # model produced JSON that failed the schema
                    last_error = "model output failed schema validation"
                    continue
                if self._downgrade(body):
                    last_error = f"HTTP 400 (downgraded request): {body[:200]}"
                    continue
            if resp.status_code in (401, 403, 404):
                raise LLMError(self._set_fatal(resp))
            if resp.status_code >= 400:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")

            try:
                analysis = self._parse(resp.json())
            except (ValueError, KeyError, IndexError, TypeError, ValidationError) as exc:
                last_error = f"invalid structured output: {str(exc)[:200]}"
                continue

            if self.cache:
                self.cache.put(key, analysis.model_dump())
            return analysis

        raise LLMError(f"gave up after {self.max_retries + 1} attempts; last error: {last_error}")

    # -- internals --------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _payload(self, text: str) -> dict[str, Any]:
        with self._lock:
            mode, extra = self.mode, dict(self.extra_params)
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(text, include_schema=(mode == "json_object"))},
            ],
        }
        if mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "resume_analysis", "schema": ResumeAnalysis.model_json_schema()},
            }
        else:
            payload["response_format"] = {"type": "json_object"}
        payload.update(extra)
        return payload

    def _set_fatal(self, resp: requests.Response) -> str:
        """A bad key or model name will not fix itself: stop calling the API and say why, once."""
        hint = {401: f"invalid or missing key: check {self.api_key_env}", 403: f"access denied: check {self.api_key_env} and plan",
                404: f"unknown model or endpoint: check llm.model ('{self.model}') and llm.base_url"}[resp.status_code]
        message = f"HTTP {resp.status_code} ({hint}): {resp.text[:200]}. LLM disabled for the rest of the run."
        with self._lock:
            if self._fatal is None:
                self._fatal = message
                log.error("LLM configuration error - %s", message)
        return message

    def _downgrade(self, error_body: str) -> bool:
        """React to a 400 by simplifying the request once. Returns True if something changed."""
        lowered = error_body.lower()
        with self._lock:
            for param in list(self.extra_params):
                if param.lower() in lowered:
                    log.warning("LLM API rejected '%s'; dropping it for the rest of the run", param)
                    self.extra_params.pop(param)
                    return True
            if self.mode == "json_schema" and ("response_format" in lowered or "json_schema" in lowered
                                               or "schema" in lowered):
                log.warning("Model does not accept json_schema; switching to json_object mode")
                self.mode = "json_object"
                return True
            if self.extra_params:  # unknown 400: try once more without extras
                self.extra_params.clear()
                return True
        return False

    @staticmethod
    def _parse(body: dict[str, Any]) -> ResumeAnalysis:
        content = body["choices"][0]["message"]["content"] or ""
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no JSON object in response")
        return ResumeAnalysis.model_validate(json.loads(content[start : end + 1]))

    def _sleep(self, attempt: int, retry_after: str | None) -> None:
        if attempt >= self.max_retries:
            return
        try:
            wait = float(retry_after) if retry_after else 2.0 * (2**attempt)
        except ValueError:
            wait = 2.0 * (2**attempt)
        wait = min(max(wait, 1.0), self.max_wait)
        log.info("LLM call throttled/failed; retrying in %.0fs", wait)
        time.sleep(wait)
