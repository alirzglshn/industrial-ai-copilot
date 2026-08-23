from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: str
    page_count: int
    uploaded_at: datetime


class DocumentUploadResponse(BaseModel):
    id: str
    filename: str
    status: str
    page_count: int
    chunk_count: int
    image_count: int


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    page_number: int
    chunk_index: int
    text: str


class ImageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    page_number: int
    image_index: int
    storage_path: str
    caption: str | None = None
