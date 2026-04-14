"""
Document parsing service.
Supports PDF (via PyMuPDF) and plain TXT files.
Extend this module to add more formats (DOCX, HTML, etc.)
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def parse_document(file_path: str) -> str:
    """
    Extract raw text from a document.
    Returns the full text as a single string.
    Raises ValueError for unsupported formats.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    if ext == ".txt":
        return _parse_txt(path)
    elif ext == ".pdf":
        return _parse_pdf(path)


def _parse_txt(path: Path) -> str:
    """Read plain text file with UTF-8 encoding, fallback to latin-1."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decode failed for {path.name}, falling back to latin-1")
        return path.read_text(encoding="latin-1")


def _parse_pdf(path: Path) -> str:
    """
    Extract text from PDF using PyMuPDF (fitz).
    Iterates all pages and joins text with newlines.
    Note: For scanned/image PDFs, OCR would be needed (not implemented here).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF not installed. Run: pip install PyMuPDF"
        )

    doc = fitz.open(str(path))
    pages_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages_text.append(f"[Page {page_num + 1}]\n{text}")

    doc.close()

    if not pages_text:
        raise ValueError(
            f"No extractable text found in '{path.name}'. "
            "The PDF may be image-based and require OCR."
        )

    return "\n\n".join(pages_text)
