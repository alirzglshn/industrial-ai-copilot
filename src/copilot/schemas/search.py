from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    # restricting the search to one manual, via a qdrant payload filter
    document_id: str | None = None
    # text-only search when false, otherwise clip and page-context images are fused in
    include_images: bool = True


class EvidenceOut(BaseModel):
    kind: str
    document_id: str
    page_number: int
    # similarity from the source model, comparable within a kind, not across kinds
    score: float
    # what determined the position when text and image rankings were fused, else null
    fused_score: float | None = None
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
    indexed_images: int
    status: str
