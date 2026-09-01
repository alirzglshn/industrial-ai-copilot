"""pdf to structured document, text, tables and images per page"""

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
    """turning a pdf file into a parsed document, one page at a time"""

    @abstractmethod
    def parse(self, file_path: str, document_id: str) -> ParsedDocument: ...
