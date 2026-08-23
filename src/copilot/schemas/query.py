from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    document_id: str | None = None
    top_k: int = 5


class Citation(BaseModel):
    document_id: str
    page_number: int
    chunk_id: str | None = None
    image_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_evidence: bool = False
