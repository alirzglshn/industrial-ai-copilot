import logging

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from copilot.core.config import Settings, get_settings
from copilot.db.models import Chunk, Document, Image
from copilot.db.session import get_db
from copilot.ingestion.service import IngestionService, get_ingestion_service
from copilot.schemas.documents import (
    ChunkOut,
    DocumentSummary,
    DocumentUploadResponse,
    ImageOut,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


def _get_document_or_404(document_id: str, db: Session) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
def upload_document(
    file: UploadFile,
    db: Session = Depends(get_db),
    service: IngestionService = Depends(get_ingestion_service),
    settings: Settings = Depends(get_settings),
) -> DocumentUploadResponse:
    """Ingests a PDF: parses text/tables/images per page, chunks it, and persists it."""
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_mb} MB limit",
        )

    try:
        document = service.ingest(db, filename, content)
    except Exception:
        # ingest() has already marked the document failed and logged the cause.
        raise HTTPException(status_code=422, detail="Could not parse the uploaded PDF")

    chunk_count = db.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == document.id)
    )
    image_count = db.scalar(
        select(func.count()).select_from(Image).where(Image.document_id == document.id)
    )

    return DocumentUploadResponse(
        id=document.id,
        filename=document.filename,
        status=document.status,
        page_count=document.page_count,
        chunk_count=chunk_count or 0,
        image_count=image_count or 0,
    )


@router.get("", response_model=list[DocumentSummary])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:
    return list(db.scalars(select(Document).order_by(Document.uploaded_at.desc())))


@router.get("/{document_id}", response_model=DocumentSummary)
def get_document(document_id: str, db: Session = Depends(get_db)) -> Document:
    return _get_document_or_404(document_id, db)


@router.get("/{document_id}/chunks", response_model=list[ChunkOut])
def list_chunks(
    document_id: str,
    page_number: int | None = Query(default=None, ge=1),
    db: Session = Depends(get_db),
) -> list[Chunk]:
    _get_document_or_404(document_id, db)

    statement = select(Chunk).where(Chunk.document_id == document_id)
    if page_number is not None:
        statement = statement.where(Chunk.page_number == page_number)
    return list(db.scalars(statement.order_by(Chunk.chunk_index)))


@router.get("/{document_id}/images", response_model=list[ImageOut])
def list_images(
    document_id: str,
    page_number: int | None = Query(default=None, ge=1),
    db: Session = Depends(get_db),
) -> list[Image]:
    _get_document_or_404(document_id, db)

    statement = select(Image).where(Image.document_id == document_id)
    if page_number is not None:
        statement = statement.where(Image.page_number == page_number)
    return list(db.scalars(statement.order_by(Image.page_number, Image.image_index)))
