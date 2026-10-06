import json
import shutil

from conftest import FIXTURES, make_pdf

from screener.pipeline import run_pipeline
from screener.report import write_outputs


def test_batch_survives_bad_files_and_reports_everything(cfg, tmp_path):
    folder = tmp_path / "resumes"
    folder.mkdir()
    for txt in FIXTURES.glob("*.txt"):
        make_pdf(folder / f"{txt.stem}.pdf", txt.read_text(encoding="utf-8"))
    shutil.copy(folder / "strong_agentic.pdf", folder / "strong_agentic_again.pdf")   # duplicate
    (folder / "broken.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")         # corrupted
    make_pdf(folder / "scanned.pdf", " ")                                               # no text
    (folder / "notes.txt").write_text((FIXTURES / "thin_wrapper.txt").read_text().replace("neha.gupta", "neha.g2"))

    results = run_pipeline(folder, cfg, use_llm=False, use_github=False)
    s = results["summary"]
    assert s["total_files"] == 10
    assert s["failed_unreadable"] == 2
    assert s["duplicates"] == 1
    assert s["eligible"] + s["rejected"] == s["unique_candidates"]
    assert {c["file"] for c in results["rejected_candidates"]} == {"java_react_only.pdf"}

    ranked = results["ranked_candidates"]
    assert ranked[0]["candidate_name"] == "Asha Rao"
    assert [c["rank"] for c in ranked] == list(range(1, len(ranked) + 1))
    assert all(c["score_breakdown"] and c["analysis_mode"] == "rules_fallback" for c in ranked)

    json_path, csv_path = write_outputs(results, tmp_path / "out" / "results.json")
    assert json.loads(json_path.read_text())["summary"] == s
    assert csv_path.exists()
