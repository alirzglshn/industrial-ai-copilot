"""Phase 2: PDF -> structured document (text, tables, images, per page).

Implemented against in Phase 2. Defined now so the rest of the pipeline
(chunking, embedding, storage) has a stable contract to build on.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ExtractedImage:
    page_number: int
    image_index: int
    storage_path: str
    caption: str | None = None


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    tables: list[str] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)


@dataclass
class ParsedDocument:
    document_id: str
    filename: str
    pages: list[ExtractedPage]


class DocumentParser(ABC):
    """Turns a PDF file into a ParsedDocument, one ExtractedPage per page."""

    @abstractmethod
    def parse(self, file_path: str, document_id: str) -> ParsedDocument: ...
