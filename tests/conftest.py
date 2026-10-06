import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screener.config import load_config  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "resumes"


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "config.yaml")


@pytest.fixture
def resume_text():
    def _load(name: str) -> str:
        return (FIXTURES / f"{name}.txt").read_text(encoding="utf-8")
    return _load


def make_pdf(path: Path, text: str) -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    assert page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), text, fontsize=9) >= 0
    doc.save(path)
