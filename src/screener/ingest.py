"""Read resume files into plain text + hyperlinks. One bad file never stops the batch."""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .extract import split_sections

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MIN_TEXT_CHARS = 150  # below this the file is almost certainly a scanned image or empty


@dataclass
class RawDocument:
    path: Path
    file_hash: str = ""
    text: str = ""
    links: list[str] = field(default_factory=list)
    name_hint: str | None = None  # largest-font text on page 1 (PDF only); usually the name
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def discover_files(input_dir: str | Path) -> list[Path]:
    """All files under input_dir (recursive), skipping hidden files and macOS zip junk."""
    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory not found: {root}")
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and not any(part.startswith(".") or part == "__MACOSX" for part in p.relative_to(root).parts)
    ]
    return sorted(files, key=lambda p: str(p.relative_to(root)).lower())


def read_document(path: Path) -> RawDocument:
    doc = RawDocument(path=path)
    try:
        data = path.read_bytes()
        doc.file_hash = hashlib.sha256(data).hexdigest()
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            doc.error = f"Unsupported file type '{ext or 'none'}' (supported: PDF, DOCX, TXT)"
            return doc
        if ext == ".pdf":
            doc.text, doc.links, doc.name_hint = _read_pdf(path)
        elif ext == ".docx":
            doc.text, doc.links = _read_docx(path)
        else:
            doc.text = data.decode("utf-8", errors="replace")
        doc.text = normalize_text(doc.text)
        if len(doc.text.strip()) < MIN_TEXT_CHARS:
            doc.error = (
                f"No extractable text ({len(doc.text.strip())} chars) - likely a scanned/image-only "
                "file; OCR is out of scope"
            )
    except Exception as exc:  # malformed files raise all sorts of library errors
        log.warning("Could not read %s: %s", path.name, exc)
        doc.error = f"Unreadable file: {type(exc).__name__}: {exc}"[:300]
    return doc


def _read_pdf(path: Path) -> tuple[str, list[str], str | None]:
    import pymupdf

    try:
        pymupdf.TOOLS.mupdf_display_errors(False)
        pymupdf.TOOLS.mupdf_display_warnings(False)
    except AttributeError:
        pass
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError("PDF is password-protected")
        pages, links = [], []
        for page in pdf:
            pages.append(page.get_text("text"))
            # "GitHub" is often clickable text; the real URL only exists in the link annotation.
            links.extend(link["uri"] for link in page.get_links() if link.get("uri"))
        text = "\n".join(pages)
        if not _has_content_sections(text):
            # Designer templates (Canva etc.) store text out of reading order, so headings end up
            # in one block and their bodies elsewhere. Re-read in visual order and keep that if it helps.
            sorted_text = "\n".join(page.get_text("text", sort=True) for page in pdf)
            if _has_content_sections(sorted_text):
                log.info("%s: read in visual order (stored text order had no project/experience sections)", path.name)
                text = sorted_text
        name_hint = _largest_font_text(pdf[0]) if len(pdf) else None
    return text, links, name_hint


def _has_content_sections(text: str) -> bool:
    sections = split_sections(normalize_text(text))
    return bool(sections.get("projects") or sections.get("experience"))


def _largest_font_text(page) -> str | None:
    """Designed resumes often store text out of reading order; the name is reliably the largest font."""
    try:
        spans = [
            (span["size"], span["bbox"][1], span["bbox"][0], span["text"].strip())
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
            if span["text"].strip()
        ]
    except Exception:
        return None
    if not spans:
        return None
    biggest = max(s[0] for s in spans)
    top = sorted((s for s in spans if s[0] >= biggest - 0.5), key=lambda s: (round(s[1]), s[2]))
    text = " ".join(s[3] for s in top)
    return " ".join(text.split())[:60] or None


def _read_docx(path: Path) -> tuple[str, list[str]]:
    import docx

    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    links = [
        rel.target_ref
        for rel in document.part.rels.values()
        if rel.reltype.endswith("/hyperlink") and rel.is_external
    ]
    return "\n".join(parts), links


_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


def normalize_text(text: str) -> str:
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\(cid:\d+\)", " ", text)
    text = re.sub("[\\u200b\\u200c\\u200d\\u2060\\ufeff]", "", text)  # zero-width characters
    text = text.replace(" ", " ").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    # collapse runs of blank lines
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


def text_fingerprint(text: str) -> str:
    """Hash of normalised text: catches the same resume saved as two different files."""
    squashed = re.sub(r"[^a-z0-9]", "", text.lower())
    return hashlib.sha256(squashed.encode()).hexdigest()
