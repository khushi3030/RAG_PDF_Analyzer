"""
pdf_processor.py
-----------------
Extracts text from a PDF, page by page, using PyMuPDF (fitz).

PyMuPDF is used over PyPDF2 because it preserves reading order much better
on multi-column pages and is significantly faster on large documents.
"""

from typing import List, Dict
import pymupdf


def extract_pages(pdf_path: str, source_name: str = None) -> List[Dict]:
    """
    Returns a list of {"text": str, "page_number": int, "source": str}
    one entry per PDF page. Empty pages (e.g. pure-image pages with no
    extractable text) are skipped.
    """
    source_name = source_name or pdf_path.split("/")[-1]
    pages = []
    doc = pymupdf.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            text = _clean_text(text)
            if text.strip():
                pages.append(
                    {
                        "text": text,
                        "page_number": i + 1,
                        "source": source_name,
                    }
                )
    finally:
        doc.close()
    return pages


def _clean_text(text: str) -> str:
    """Collapse excessive whitespace/hyphenation artifacts left by PDF extraction."""
    # de-hyphenate words split across a line break, e.g. "infor-\nmation"
    text = text.replace("-\n", "")
    # normalize remaining newlines/spaces without losing paragraph breaks
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln != ""]
    return "\n".join(lines)