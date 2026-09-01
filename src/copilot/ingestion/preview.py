"""rendering one pdf page as an image, for the citation's source-page view"""

import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)


class PagePreviewError(ValueError):
    pass


def render_page(pdf_path: Path, page_number: int, cache_dir: Path, resolution: int = 150) -> Path:
    """path to a cached png of the given page, rendering it first if needed"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"page{page_number:04d}.png"
    if cached.exists():
        return cached

    if not pdf_path.exists():
        raise PagePreviewError(f"Source PDF not found: {pdf_path}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not 1 <= page_number <= len(pdf.pages):
                raise PagePreviewError(
                    f"Page {page_number} out of range (document has {len(pdf.pages)} pages)"
                )
            image = pdf.pages[page_number - 1].to_image(resolution=resolution)
            image.save(cached)
    except PagePreviewError:
        raise
    except Exception as error:
        logger.warning("Could not render page %s of %s: %s", page_number, pdf_path, error)
        raise PagePreviewError(f"Could not render page {page_number}") from error

    return cached
