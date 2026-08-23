from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    # Restricts the search to one manual, via a Qdrant payload filter.
    document_id: str | None = None


class EvidenceOut(BaseModel):
    kind: str
    document_id: str
    page_number: int
    score: float
    chunk_id: str | None = None
    text: str | None = None
    image_id: str | None = None
    image_path: str | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[EvidenceOut]


class IndexResponse(BaseModel):
    document_id: str
    indexed_chunks: int
    status: str
