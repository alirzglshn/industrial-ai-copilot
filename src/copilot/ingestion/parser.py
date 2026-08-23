"""Phase 2: PDF -> ParsedDocument (text, tables, images, per page).

Two libraries are used deliberately:

- **pdfplumber** (on pdfminer.six) for text and table extraction. Its table
  detection is the reason it is here; technical manuals put specifications
  and tolerances in tables, and losing them to flattened prose would make
  exactly the questions this system targets unanswerable.
- **pypdf** for pulling embedded raster images (diagrams, schematics,
  photos) out of each page.

Both are permissively licensed and pure-Python, so ingestion needs no
system packages and runs the same on a laptop and in the API container.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

from copilot.ingestion.base import (
    DocumentParser,
    ExtractedImage,
    ExtractedPage,
    ParsedDocument,
)

logger = logging.getLogger(__name__)


@dataclass
class PdfParserConfig:
    image_dir: Path
    # Manuals repeat logos, rules, and spacer graphics on every page. Indexing
    # those buries real diagrams in the image search results, so anything
    # smaller than this in either dimension is skipped.
    min_image_width: int = 64
    min_image_height: int = 64


def _serialize_table(table: list[list[str | None]]) -> str:
    """Render an extracted table as pipe-separated rows.

    Kept as text rather than a structured type because the downstream
    consumers (the embedding model and the answering VLM) both take text, and
    a pipe-separated grid preserves row/column adjacency that flattened prose
    would lose.
    """
    lines = []
    for row in table:
        cells = [(cell or "").replace("\n", " ").strip() for cell in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


class PdfDocumentParser(DocumentParser):
    def __init__(self, config: PdfParserConfig) -> None:
        self.config = config

    def parse(self, file_path: str, document_id: str) -> ParsedDocument:
        path = Path(file_path)
        images_by_page = self._extract_images(path, document_id)

        pages: list[ExtractedPage] = []
        with pdfplumber.open(path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                found_tables = []
                tables: list[str] = []
                try:
                    found_tables = page.find_tables()
                    for table in found_tables:
                        serialized = _serialize_table(table.extract())
                        if serialized:
                            tables.append(serialized)
                except Exception:
                    # A single unparseable table must not cost us the page's text.
                    logger.warning(
                        "Table extraction failed on page %s of %s", page_number, path.name
                    )
                    found_tables = []

                text = self._text_outside_tables(page, found_tables, page_number, path.name)

                pages.append(
                    ExtractedPage(
                        page_number=page_number,
                        text=text,
                        tables=tables,
                        images=images_by_page.get(page_number, []),
                    )
                )

        return ParsedDocument(document_id=document_id, filename=path.name, pages=pages)

    @staticmethod
    def _text_outside_tables(page, tables, page_number: int, filename: str) -> str:
        """Page text with table regions removed.

        pdfplumber's extract_text() also returns the words inside tables, so
        without this a table's contents land in both the prose chunk and the
        dedicated [Table] chunk. Duplicated evidence is actively harmful: two
        near-identical chunks can occupy two of the top-k retrieval slots and
        crowd out genuinely different evidence.
        """
        if not tables:
            return page.extract_text() or ""

        boxes = [table.bbox for table in tables]

        def outside_every_table(obj) -> bool:
            center_x = (obj["x0"] + obj["x1"]) / 2
            center_y = (obj["top"] + obj["bottom"]) / 2
            return not any(
                x0 <= center_x <= x1 and top <= center_y <= bottom
                for x0, top, x1, bottom in boxes
            )

        try:
            return page.filter(outside_every_table).extract_text() or ""
        except Exception:
            # Falling back to the full text duplicates the table rather than
            # losing the page entirely, which is the safer failure.
            logger.warning(
                "Could not exclude table regions on page %s of %s", page_number, filename
            )
            return page.extract_text() or ""

    def _extract_images(self, path: Path, document_id: str) -> dict[int, list[ExtractedImage]]:
        output_dir = self.config.image_dir / document_id
        output_dir.mkdir(parents=True, exist_ok=True)

        images_by_page: dict[int, list[ExtractedImage]] = {}
        reader = PdfReader(str(path))

        for page_number, page in enumerate(reader.pages, start=1):
            extracted: list[ExtractedImage] = []
            try:
                page_images = list(page.images)
            except Exception:
                # Exotic filters and malformed XObjects are common in the wild;
                # a page whose images cannot be read still contributes its text.
                logger.warning("Image extraction failed on page %s of %s", page_number, path.name)
                page_images = []

            for image_index, image_file in enumerate(page_images):
                try:
                    image = image_file.image
                except Exception:
                    logger.warning(
                        "Could not decode image %s on page %s of %s",
                        image_index,
                        page_number,
                        path.name,
                    )
                    continue

                if image is None:
                    continue
                if (
                    image.width < self.config.min_image_width
                    or image.height < self.config.min_image_height
                ):
                    continue

                storage_path = output_dir / f"page{page_number:04d}_img{image_index:02d}.png"
                try:
                    # Normalize everything to PNG so downstream image embedding
                    # and page previews deal with exactly one format.
                    image.convert("RGB").save(storage_path, format="PNG")
                except Exception:
                    logger.warning("Could not save image %s from page %s", image_index, page_number)
                    continue

                extracted.append(
                    ExtractedImage(
                        page_number=page_number,
                        image_index=image_index,
                        storage_path=str(storage_path),
                    )
                )

            if extracted:
                images_by_page[page_number] = extracted

        return images_by_page
