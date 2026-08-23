"""Phase 2: orchestration — upload -> parse -> chunk -> persist.

This is the seam later phases plug into: Phase 3 embeds the Chunk rows this
writes, and Phase 4 embeds the Image rows. Nothing here talks to a model, so
ingestion stays runnable (and testable) without any ML dependencies installed.
"""

import logging
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from copilot.core.config import Settings, get_settings
from copilot.db.models import Chunk, Document, Image, Page
from copilot.ingestion.base import DocumentParser
from copilot.ingestion.chunker import TextChunker
from copilot.ingestion.parser import PdfDocumentParser, PdfParserConfig

logger = logging.getLogger(__name__)

STATUS_PARSING = "parsing"
STATUS_PARSED = "parsed"
STATUS_FAILED = "failed"


class IngestionService:
    def __init__(self, parser: DocumentParser, chunker: TextChunker, upload_dir: Path) -> None:
        self.parser = parser
        self.chunker = chunker
        self.upload_dir = upload_dir

    def save_upload(self, document_id: str, filename: str, content: bytes) -> Path:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        # Stored under the document id, not the original filename: two manuals
        # can share a name, and the id is what every downstream row references.
        path = self.upload_dir / f"{document_id}.pdf"
        path.write_bytes(content)
        return path

    def ingest(self, db: Session, filename: str, content: bytes) -> Document:
        document_id = str(uuid.uuid4())
        path = self.save_upload(document_id, filename, content)

        document = Document(id=document_id, filename=filename, status=STATUS_PARSING)
        db.add(document)
        db.commit()

        try:
            parsed = self.parser.parse(str(path), document_id)
        except Exception:
            logger.exception("Parsing failed for document %s (%s)", document_id, filename)
            document.status = STATUS_FAILED
            db.commit()
            raise

        for page in parsed.pages:
            db.add(Page(document_id=document_id, page_number=page.page_number))
            for image in page.images:
                db.add(
                    Image(
                        document_id=document_id,
                        page_number=image.page_number,
                        image_index=image.image_index,
                        storage_path=image.storage_path,
                        caption=image.caption,
                    )
                )

        for chunk in self.chunker.chunk_document(parsed):
            db.add(
                Chunk(
                    document_id=document_id,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                )
            )

        document.page_count = len(parsed.pages)
        document.status = STATUS_PARSED
        db.commit()
        db.refresh(document)
        return document


def build_ingestion_service(settings: Settings | None = None) -> IngestionService:
    settings = settings or get_settings()
    parser = PdfDocumentParser(PdfParserConfig(image_dir=Path(settings.image_dir)))
    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return IngestionService(parser=parser, chunker=chunker, upload_dir=Path(settings.upload_dir))


def get_ingestion_service() -> IngestionService:
    """FastAPI dependency. Overridden in tests to redirect storage to a tmp dir."""
    return build_ingestion_service()
