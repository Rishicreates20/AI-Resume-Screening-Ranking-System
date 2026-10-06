"""Measure the speed-up from bounded concurrency in the LLM and GitHub stages.

Network latency is SIMULATED (a sleep per call) so the benchmark is deterministic, free, and does
not burn API quota. It therefore measures the pipeline's scheduling, not Groq's or GitHub's speed:
with real calls, the achievable speed-up is capped by the provider's rate limits.

    python scripts/benchmark.py --input ./resumes
    python scripts/benchmark.py --input ./resumes --llm-latency 0.5 --github-latency 0.3 --workers 1 2 4 8
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screener.config import load_config  # noqa: E402
from screener.llm.rules_fallback import RulesAnalyzer  # noqa: E402
from screener.models import GitHubResult, ResumeAnalysis  # noqa: E402
from screener.pipeline import run_pipeline  # noqa: E402


class SimulatedLLM:
    """Rules analysis plus a fixed delay, standing in for an LLM round trip."""

    name = "simulated-llm"

    def __init__(self, cfg: dict, latency: float):
        self._rules, self._latency = RulesAnalyzer(cfg), latency

    def analyze(self, resume_text: str) -> ResumeAnalysis:
        time.sleep(self._latency)
        return self._rules.analyze(resume_text)


class SimulatedGitHub:
    token = None

    def __init__(self, latency: float):
        self._latency = latency

    def enrich(self, username: str | None, now: datetime | None = None) -> GitHubResult:
        if not username:
            return GitHubResult(status="no_profile", summary="No GitHub profile on resume")
        time.sleep(self._latency)
        return GitHubResult(status="ok", username=username, summary="simulated")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(ROOT / "resumes"))
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--llm-latency", type=float, default=0.3, help="simulated seconds per LLM call")
    ap.add_argument("--github-latency", type=float, default=0.2, help="simulated seconds per GitHub call")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)

    base_cfg = load_config(args.config)
    rows = []
    for n in args.workers:
        cfg = copy.deepcopy(base_cfg)
        cfg["llm"]["max_concurrency"] = n
        cfg["github"]["max_concurrency"] = n
        res = run_pipeline(args.input, cfg, llm=SimulatedLLM(cfg, args.llm_latency),
                           github=SimulatedGitHub(args.github_latency))
        st = res["summary"]["stage_seconds"]
        rows.append((n, st["analysis"], st["github"], st["analysis"] + st["github"], res["summary"]["eligible"]))

    base = rows[0][3]
    print(f"\nSimulated latency: {args.llm_latency}s per LLM call, {args.github_latency}s per GitHub call "
          f"({rows[0][4]} eligible candidates)\n")
    print("| workers | LLM stage (s) | GitHub stage (s) | total (s) | speed-up |")
    print("|---:|---:|---:|---:|---:|")
    for n, llm_s, gh_s, total, _ in rows:
        print(f"| {n} | {llm_s:.2f} | {gh_s:.2f} | {total:.2f} | {base / total:.1f}x |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
