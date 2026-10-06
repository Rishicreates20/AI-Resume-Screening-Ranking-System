"""CLI entry point.

    python main.py --input ./resumes --output ./output/results.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from screener.config import load_config  # noqa: E402
from screener.pipeline import run_pipeline  # noqa: E402
from screener.report import print_report, write_outputs  # noqa: E402
from screener.report_html import write_html  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Screen and rank a folder of resumes.")
    parser.add_argument("--input", default="./resumes", help="Folder of resumes (PDF/DOCX/TXT)")
    parser.add_argument("--output", default="./output/results.json", help="Path of the JSON results file")
    parser.add_argument("--config", default="config.yaml", help="Config file (weights, terms, model)")
    parser.add_argument("--no-llm", action="store_true", help="Rules-only analysis (no API calls)")
    parser.add_argument("--no-github", action="store_true", help="Skip GitHub enrichment")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Process only the first N files (cheap trial run before a long LLM batch)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    # requests/urllib3 are chatty at DEBUG; keep our own logs readable
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    try:
        cfg = load_config(args.config)
        results = run_pipeline(args.input, cfg, use_llm=not args.no_llm, use_github=not args.no_github,
                               limit=args.limit)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logging.error("%s", exc)
        return 2

    json_path, csv_path = write_outputs(results, args.output)
    html_path = write_html(results, json_path.with_suffix(".html"))
    print_report(results, cfg.get("output", {}).get("top_n_console", 15))
    print(f"Wrote {json_path}, {csv_path} and {html_path} (open the HTML for the dashboard)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
