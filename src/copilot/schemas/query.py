from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    # whether diagrams may be retrieved as evidence alongside passages
    include_images: bool = True
    # continuing an existing conversation's history, affects only what gets logged
    conversation_id: str | None = None


class Citation(BaseModel):
    kind: str
    document_id: str
    page_number: int
    chunk_id: str | None = None
    image_id: str | None = None
    image_path: str | None = None


class QueryResponse(BaseModel):
    answer: str
    # evidence the answer actually cited, not everything retrieved
    citations: list[Citation]
    # true when the model declined for lack of evidence, or nothing was retrieved
    insufficient_evidence: bool = False
    # pages the answer cited that were not in the retrieved evidence
    unsupported_pages: list[int] = []
    # false when the answer's claim is not backed by its cited evidence
    grounded: bool = True
    # share of the answer's content words found in the evidence it cited
    faithfulness: float = 1.0
    # log of tool calls made to produce this answer, empty for the fixed /query pipeline
    tool_calls: list[str] = []
    # the conversation this exchange was logged under, pass back to keep a thread grouped
    conversation_id: str | None = None
