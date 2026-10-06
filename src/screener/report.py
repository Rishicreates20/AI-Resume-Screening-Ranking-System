"""Write results.json + a flat CSV, and print a short console report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _csv_safe(value: Any) -> Any:
    """Resume text lands in cells that recruiters open in Excel: a name like =HYPERLINK(...) must stay text."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", chr(9), chr(13)):
        return "'" + value
    return value


def write_outputs(results: dict[str, Any], json_path: str | Path) -> tuple[Path, Path]:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = json_path.with_suffix(".csv")
    fields = ["rank", "candidate_name", "status", "total_score", "ai_project_depth", "python_backend",
              "cloud_fullstack", "github", "engineering_depth", "penalties", "email", "github_url",
              "analysis_mode", "reasons", "file"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        def put(row: dict[str, Any]) -> None:
            writer.writerow({k: _csv_safe(v) for k, v in row.items()})

        for c in results["ranked_candidates"]:
            b = c["score_breakdown"]
            put({
                "rank": c["rank"], "candidate_name": c["candidate_name"], "status": c["status"],
                "total_score": c["total_score"], **{k: b[k] for k in b}, "email": c["email"],
                "github_url": c["github_url"], "analysis_mode": c["analysis_mode"],
                "reasons": " | ".join(c["strengths"][:2] + c["concerns"][:2]), "file": c["file"],
            })
        for c in results["rejected_candidates"]:
            put({"candidate_name": c["candidate_name"], "status": "rejected", "email": c.get("email"),
                             "reasons": " | ".join(c["rejection_reasons"]), "file": c["file"]})
        for c in results["failed_files"] + results["duplicates"]:
            put({"candidate_name": c["candidate_name"], "status": c["status"],
                             "reasons": c.get("error", ""), "file": c["file"]})
    return json_path, csv_path


def print_report(results: dict[str, Any], top_n: int = 15) -> None:
    s = results["summary"]
    print()
    print("=" * 100)
    print(f"Resumes: {s['total_files']} files | parsed {s['parsed_successfully']} | failed {s['failed_unreadable']} | "
          f"duplicates {s['duplicates']} | eligible {s['eligible']} | rejected {s['rejected']}")
    print(f"LLM: {results['run']['llm']} | analysis modes {s['analysis_mode']} | GitHub: {results['run']['github']} "
          f"{s['github_enrichment']}")
    print("=" * 100)
    modes = s["analysis_mode"]
    if results["run"].get("llm_enabled") and modes["rules_fallback"]:
        print(f"WARNING: {modes['rules_fallback']} resume(s) were scored with the rules fallback because the LLM call failed "
              f"(see analysis_notes in results.json).")
    header = f"{'#':>3}  {'Candidate':<26}{'Total':>6}{'AI':>6}{'Py':>6}{'Cloud':>6}{'GH':>5}{'Eng':>5}{'Pen':>6}  Top reason"
    print(header)
    print("-" * 100)
    for c in results["ranked_candidates"][:top_n]:
        b = c["score_breakdown"]
        reason = (c["strengths"] or c["concerns"] or [""])[0]
        print(f"{c['rank']:>3}  {c['candidate_name'][:25]:<26}{c['total_score']:>6}{b['ai_project_depth']:>6}"
              f"{b['python_backend']:>6}{b['cloud_fullstack']:>6}{b['github']:>5}{b['engineering_depth']:>5}"
              f"{b['penalties']:>6}  {reason[:40]}")
    if results["rejected_candidates"]:
        print("-" * 100)
        print("Rejected:")
        for c in results["rejected_candidates"]:
            print(f"  - {c['candidate_name'][:30]:<31} {'; '.join(c['rejection_reasons'])[:65]}")
    for c in results["failed_files"] + results["duplicates"]:
        print(f"  ! {c['file'][:30]:<31} {c.get('error', '')[:65]}")
    print()
