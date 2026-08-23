from fastapi import APIRouter, HTTPException

from copilot.schemas.query import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse, status_code=501)
def ask(request: QueryRequest) -> QueryResponse:
    """Answers a question about an uploaded manual, with page citations.

    Not implemented until retrieval (Phase 3-4) and grounded generation
    (Phase 5) land. See copilot.retrieval.base.Retriever and
    copilot.generation.base.AnswerGenerator.
    """
    raise HTTPException(status_code=501, detail="Retrieval and generation land in Phase 3-5")
