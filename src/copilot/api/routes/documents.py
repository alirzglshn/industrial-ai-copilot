import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from copilot.api.deps import get_optional_retrieval_stack, require_retrieval_stack
from copilot.core.config import Settings, get_settings
from copilot.db.models import Chunk, Document, Image
from copilot.db.session import get_db
from copilot.ingestion.preview import PagePreviewError, render_page
from copilot.ingestion.service import IngestionService, get_ingestion_service
from copilot.retrieval.deps import RetrievalStack
from copilot.schemas.documents import (
    ChunkOut,
    DocumentSummary,
    DocumentUploadResponse,
    ImageOut,
)
from copilot.schemas.search import IndexResponse

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
    stack: RetrievalStack | None = Depends(get_optional_retrieval_stack),
) -> DocumentUploadResponse:
    """parsing and persisting a pdf, page by page"""
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
        # ingest already marked the document failed and logged the cause
        raise HTTPException(status_code=422, detail="Could not parse the uploaded PDF")

    chunk_count = db.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == document.id)
    )
    image_count = db.scalar(
        select(func.count()).select_from(Image).where(Image.document_id == document.id)
    )

    indexed_chunks = 0
    indexed_images = 0
    if settings.auto_index_on_upload:
        if stack is None:
            # indexing skipped, can be retried later via /documents/{id}/index
            logger.warning("Skipped indexing document %s: retrieval unavailable", document.id)
        else:
            indexed_chunks = stack.indexer.index_document(db, document.id)
            indexed_images = _index_images(stack, db, document.id)

    return DocumentUploadResponse(
        id=document.id,
        filename=document.filename,
        status=document.status,
        page_count=document.page_count,
        chunk_count=chunk_count or 0,
        image_count=image_count or 0,
        indexed_chunks=indexed_chunks,
        indexed_images=indexed_images,
    )


def _index_images(stack: RetrievalStack, db: Session, document_id: str) -> int:
    """embedding a document's images, tolerating an unavailable image model"""
    if stack.image_indexer is None:
        return 0
    try:
        return stack.image_indexer.index_document(db, document_id)
    except Exception:
        logger.warning("Image indexing failed for document %s", document_id, exc_info=True)
        return 0


@router.post("/{document_id}/index", response_model=IndexResponse)
def index_document(
    document_id: str,
    db: Session = Depends(get_db),
    stack: RetrievalStack = Depends(require_retrieval_stack),
) -> IndexResponse:
    """re-embedding a document's chunks, safe to call repeatedly"""
    document = _get_document_or_404(document_id, db)
    indexed = stack.indexer.index_document(db, document_id)
    indexed_images = _index_images(stack, db, document_id)
    db.refresh(document)
    return IndexResponse(
        document_id=document_id,
        indexed_chunks=indexed,
        indexed_images=indexed_images,
        status=document.status,
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


@router.get("/{document_id}/images/{image_id}/file")
def get_image_file(document_id: str, image_id: str, db: Session = Depends(get_db)) -> FileResponse:
    """serving an extracted diagram's actual image bytes"""
    image = db.get(Image, image_id)
    if image is None or image.document_id != document_id:
        raise HTTPException(status_code=404, detail="Image not found")

    path = Path(image.storage_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file missing on disk")
    return FileResponse(path, media_type="image/png")


@router.get("/{document_id}/pages/{page_number}/preview")
def get_page_preview(
    document_id: str,
    page_number: int,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """rendering and caching a source pdf page as png"""
    document = _get_document_or_404(document_id, db)
    if not 1 <= page_number <= document.page_count:
        raise HTTPException(
            status_code=404,
            detail=f"Page {page_number} out of range (document has {document.page_count} pages)",
        )

    pdf_path = Path(settings.upload_dir) / f"{document_id}.pdf"
    cache_dir = Path(settings.preview_dir) / document_id
    try:
        png_path = render_page(pdf_path, page_number, cache_dir)
    except PagePreviewError as error:
        raise HTTPException(status_code=404, detail=str(error))

    return FileResponse(png_path, media_type="image/png")
