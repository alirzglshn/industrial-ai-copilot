"""pdf to text, tables, and images per page, via pdfplumber and pypdf"""

import logging
import re
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

# unresolvable glyph ids from fonts with no ToUnicode map, stripped rather than indexed as noise
_CID_TOKEN = re.compile(r"\(cid:\d+\)")


@dataclass
class PdfParserConfig:
    image_dir: Path
    # skipping logos, rules, and spacer graphics repeated on every page
    min_image_width: int = 64
    min_image_height: int = 64
    # table quality gate, since ruled diagram frames and chart axes look like tables too
    min_table_rows: int = 2
    min_table_cols: int = 2
    min_table_filled_cells: int = 4
    min_table_fill_ratio: float = 0.3
    min_table_alpha_chars: int = 8
    # a letterless grid can still count as a table with enough populated cells and rows
    numeric_table_filled_cells: int = 8
    min_numeric_table_rows: int = 3


def clean_extracted_text(text: str) -> str:
    """stripping unresolvable glyph ids, normalizing whitespace, keeping paragraph breaks"""
    lines = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            lines.append("")
            continue
        line = _CID_TOKEN.sub("", raw_line)
        lines.append(re.sub(r"[ \t]+", " ", line).strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _clean_cells(table: list[list[str | None]]) -> list[list[str]]:
    return [
        [re.sub(r"\s+", " ", _CID_TOKEN.sub("", cell or "")).strip() for cell in row]
        for row in table
    ]


def is_tabular(table: list[list[str | None]], config: PdfParserConfig) -> bool:
    """whether a detected grid carries enough real content to count as a table"""
    rows = _clean_cells(table)
    if len(rows) < config.min_table_rows:
        return False
    if max((len(row) for row in rows), default=0) < config.min_table_cols:
        return False

    cells = [cell for row in rows for cell in row]
    filled = [cell for cell in cells if cell]
    if len(filled) < config.min_table_filled_cells:
        return False
    if len(filled) / len(cells) < config.min_table_fill_ratio:
        return False

    alpha_chars = sum(character.isalpha() for cell in filled for character in cell)
    if alpha_chars >= config.min_table_alpha_chars:
        return True
    return (
        len(filled) >= config.numeric_table_filled_cells
        and len(rows) >= config.min_numeric_table_rows
    )


def _serialize_table(table: list[list[str | None]]) -> str:
    """rendering an extracted table as pipe-separated rows, preserving row and column adjacency"""
    lines = []
    for row in _clean_cells(table):
        if any(row):
            lines.append(" | ".join(row))
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
                kept = self._extract_tables(page, page_number, path.name)

                # removing only kept table regions from the page text, leaving rejected grids in prose
                text = self._text_outside_tables(
                    page, [table for table, _ in kept], page_number, path.name
                )

                pages.append(
                    ExtractedPage(
                        page_number=page_number,
                        text=clean_extracted_text(text),
                        tables=[serialized for _, serialized in kept],
                        images=images_by_page.get(page_number, []),
                    )
                )

        return ParsedDocument(document_id=document_id, filename=path.name, pages=pages)

    def _extract_tables(self, page, page_number: int, filename: str) -> list[tuple[object, str]]:
        """detected grids that pass the quality gate, paired with their text"""
        try:
            found = page.find_tables()
        except Exception:
            # a page whose tables cannot be detected still contributes its text
            logger.warning("Table detection failed on page %s of %s", page_number, filename)
            return []

        kept: list[tuple[object, str]] = []
        for table in found:
            try:
                rows = table.extract()
            except Exception:
                logger.warning(
                    "Could not read a detected table on page %s of %s", page_number, filename
                )
                continue

            if not is_tabular(rows, self.config):
                continue
            serialized = _serialize_table(rows)
            if serialized:
                kept.append((table, serialized))

        return kept

    @staticmethod
    def _text_outside_tables(page, tables, page_number: int, filename: str) -> str:
        """page text with table regions removed, avoiding duplicate evidence in retrieval"""
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
            # falling back to the full text duplicates the table rather than losing the page
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
                # a page whose images cannot be read still contributes its text
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
                    # normalizing to png so downstream embedding and previews deal with one format
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
