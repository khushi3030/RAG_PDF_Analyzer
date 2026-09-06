"""
image_processor.py
-------------------
Handles the two cases where content lives in images rather than plain text:

1. Scanned pages: a page with NO extractable text (it's literally a photo
   of a page). We render the page to an image and ask a local vision model
   to transcribe it.
2. Embedded figures: a normal text page that also contains charts/diagrams/
   photos. We extract each embedded image and ask the vision model to
   describe it, so questions like "what does the chart on page 5 show"
   become answerable.

Both paths return data in the same {"text", "page_number", "source"} shape
used by pdf_processor.extract_pages(), so they merge into the SAME
chunking → embedding → retrieval pipeline with no other changes needed.
"""

import base64
from typing import List, Dict
import pymupdf
import ollama

# llama3.2-vision uses the 'mllama' architecture, which Ollama's engine
# (v0.30.0+) currently does not support — see github.com/ollama/ollama/issues/16547.
# llava is used instead: mature, widely supported, and unaffected by that issue.
VISION_MODEL = "llava"

OCR_PROMPT = (
    "Transcribe all readable text from this image exactly as it appears, "
    "preserving structure like headings and lists. If it contains a chart, "
    "diagram, or figure, also describe what it shows in 2-3 sentences."
)

FIGURE_PROMPT = (
    "Describe this figure, chart, or image in 2-4 sentences. Include any "
    "visible text, numbers, axis labels, or captions."
)


def _page_to_base64_png(page, zoom: float = 2.0) -> str:
    """Render a PDF page to a PNG image, base64-encoded, for the vision model.
    zoom=2.0 roughly doubles resolution, which meaningfully improves OCR
    accuracy on small text compared to the default render size."""
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


def describe_image(image_b64: str, prompt: str, model: str = VISION_MODEL) -> str:
    """Returns an empty string on failure instead of raising, so a vision
    model problem (wrong model name, model not pulled, architecture error,
    Ollama down, etc.) never crashes the rest of PDF processing — text
    extraction and chunking still succeed even if image analysis can't."""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"[image_processor] vision model call failed, skipping this image: {e}")
        return ""


def extract_scanned_pages(pdf_path: str, source_name: str = None, model: str = VISION_MODEL) -> List[Dict]:
    """Finds pages with no extractable text and transcribes them via vision model."""
    source_name = source_name or pdf_path.split("/")[-1]
    pages = []
    doc = pymupdf.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            if page.get_text("text").strip():
                continue  # has real text already — handled by pdf_processor.py
            img_b64 = _page_to_base64_png(page)
            transcribed = describe_image(img_b64, OCR_PROMPT, model=model)
            if transcribed.strip():
                pages.append(
                    {"text": transcribed, "page_number": i + 1, "source": source_name}
                )
    finally:
        doc.close()
    return pages


def extract_embedded_figures(pdf_path: str, source_name: str = None, model: str = VISION_MODEL) -> List[Dict]:
    """Finds embedded RASTER images (actual photo/bitmap files stuck into
    the PDF) on text pages and describes each one. Note: this does NOT
    catch vector-drawn charts (see extract_vector_figures below) — those
    are a completely different thing in the PDF's internal structure."""
    source_name = source_name or pdf_path.split("/")[-1]
    results = []
    doc = pymupdf.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                img_bytes = doc.extract_image(xref)["image"]
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                description = describe_image(img_b64, FIGURE_PROMPT, model=model)
                if description.strip():
                    results.append(
                        {
                            "text": f"[Figure {img_index + 1}, page {i + 1}]: {description}",
                            "page_number": i + 1,
                            "source": source_name,
                        }
                    )
    finally:
        doc.close()
    return results


def extract_vector_figures(
    pdf_path: str, source_name: str = None, model: str = VISION_MODEL, min_drawings: int = 3
) -> List[Dict]:
    """
    Research papers and technical documents very often draw charts/graphs/
    diagrams as VECTOR GRAPHICS (lines, curves, shapes composed
    mathematically — e.g. matplotlib/TikZ output) rather than as embedded
    bitmap images. get_images() cannot see these at all — they're a
    different kind of PDF content entirely. This function catches that
    case: if a page has a meaningful number of vector drawing operations
    (more than `min_drawings`), it's likely to contain a chart, so the
    whole page is rendered as an image and sent to the vision model, which
    is asked to describe any figure it finds or say NONE if there isn't one.
    """
    source_name = source_name or pdf_path.split("/")[-1]
    results = []
    doc = pymupdf.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            if len(page.get_drawings()) < min_drawings:
                continue
            img_b64 = _page_to_base64_png(page)
            description = describe_image(
                img_b64,
                "This page may contain a chart, graph, or diagram made of "
                "vector graphics (lines, curves, shapes). If it does, "
                "describe what it shows in 2-4 sentences, including axis "
                "labels and key data points if visible. If there is no "
                "meaningful chart or diagram on this page, reply with "
                "exactly one word: NONE",
                model=model,
            )
            cleaned = description.strip()
            if cleaned and cleaned.upper() != "NONE" and not cleaned.upper().startswith("NONE"):
                results.append(
                    {
                        "text": f"[Diagram/chart on page {i + 1}]: {cleaned}",
                        "page_number": i + 1,
                        "source": source_name,
                    }
                )
    finally:
        doc.close()
    return results