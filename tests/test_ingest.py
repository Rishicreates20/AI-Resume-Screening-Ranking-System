import pymupdf
from conftest import FIXTURES, make_pdf

from screener.extract import split_sections
from screener.ingest import discover_files, read_document


def _fixture(name):
    return (FIXTURES / f"{name}.txt").read_text(encoding="utf-8")


def test_docx_resume_is_parsed_including_hyperlinks(tmp_path):
    import docx

    d = docx.Document()
    for line in _fixture("strong_agentic").splitlines():
        d.add_paragraph(line)
    d.part.relate_to("https://github.com/asharao-dev", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                     is_external=True)
    path = tmp_path / "asha.docx"
    d.save(path)

    doc = read_document(path)
    assert doc.ok
    assert "LangGraph" in doc.text and "PROJECTS" in doc.text
    assert "https://github.com/asharao-dev" in doc.links


def test_txt_and_unsupported_and_empty_files_never_raise(tmp_path):
    (tmp_path / "a.txt").write_text(_fixture("strong_agentic"), encoding="utf-8")
    (tmp_path / "b.xyz").write_bytes(b"whatever")
    (tmp_path / "c.pdf").write_bytes(b"")
    docs = {p.name: read_document(p) for p in discover_files(tmp_path)}
    assert docs["a.txt"].ok
    assert "Unsupported file type" in docs["b.xyz"].error
    assert not docs["c.pdf"].ok


def test_password_protected_pdf_is_reported_not_crashed(tmp_path):
    make_pdf(tmp_path / "plain.pdf", _fixture("strong_agentic"))
    src = pymupdf.open(tmp_path / "plain.pdf")
    src.save(tmp_path / "locked.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    doc = read_document(tmp_path / "locked.pdf")
    assert not doc.ok and "password" in doc.error.lower()


def test_hidden_files_and_zip_junk_are_skipped(tmp_path):
    (tmp_path / "__MACOSX").mkdir()
    (tmp_path / "__MACOSX" / "._cv.pdf").write_bytes(b"junk")
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    (tmp_path / "real.txt").write_text("x" * 200)
    assert [p.name for p in discover_files(tmp_path)] == ["real.txt"]


def test_scrambled_pdf_text_order_is_repaired_by_visual_order_fallback(tmp_path):
    """Designer templates store the body text *before* its heading. Reading in stored order loses the
    PROJECTS section; the visual-order fallback must recover it."""
    pdf = pymupdf.open()
    page = pdf.new_page()
    body = ["Agentic Desk | Python, LangGraph, FastAPI",
            "- Built a multi-agent LangGraph workflow with tool calling and RAG over support documents.",
            "- Async FastAPI backend with Redis caching, deployed with Docker on Cloud Run for 40 analysts."]
    for i, line in enumerate(body):                       # stored FIRST, drawn lower on the page
        page.insert_text((40, 140 + 16 * i), line, fontsize=9)
    page.insert_text((40, 60), "Asha Rao", fontsize=18)   # stored later, drawn at the top
    page.insert_text((40, 90), "asha.rao@example.com  +91 98765 43210  github.com/asharao-dev", fontsize=9)
    page.insert_text((40, 120), "PROJECTS", fontsize=11)
    pdf.save(tmp_path / "scrambled.pdf")

    doc = read_document(tmp_path / "scrambled.pdf")
    assert doc.ok
    assert "LangGraph" in split_sections(doc.text).get("projects", "")
