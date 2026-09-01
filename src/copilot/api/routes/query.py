"""grounded multimodal question answering"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from copilot.api.deps import require_answer_generator, require_retrieval_stack
from copilot.api.sse import sse_event
from copilot.conversation.service import get_or_create_conversation, record_exchange
from copilot.db.models import Conversation
from copilot.db.session import get_db, get_session_factory
from copilot.generation.base import Answer
from copilot.retrieval.base import Evidence
from copilot.schemas.query import Citation, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


def _to_citation(evidence: Evidence) -> Citation:
    return Citation(
        kind=evidence.kind.value,
        document_id=evidence.document_id,
        page_number=evidence.page_number,
        chunk_id=evidence.chunk_id,
        image_id=evidence.image_id,
        image_path=evidence.image_path,
    )


def _to_response(question: str, answer: Answer, conversation_id: str | None = None) -> QueryResponse:
    if not answer.grounded:
        logger.warning(
            "Answer to %r is not supported by the retrieved evidence "
            "(cited %d item(s), faithfulness %.2f)",
            question,
            len(answer.evidence_used),
            answer.faithfulness,
        )

    return QueryResponse(
        answer=answer.text,
        citations=[_to_citation(item) for item in answer.evidence_used],
        insufficient_evidence=answer.insufficient_evidence,
        unsupported_pages=answer.unsupported_pages,
        grounded=answer.grounded,
        faithfulness=answer.faithfulness,
        tool_calls=answer.tool_calls,
        conversation_id=conversation_id,
    )


@router.post("/query", response_model=QueryResponse)
def ask(
    request: QueryRequest,
    db: Session = Depends(get_db),
    stack=Depends(require_retrieval_stack),
    generator=Depends(require_answer_generator),
) -> QueryResponse:
    """answering a question about an uploaded manual, citing the pages used"""
    evidence = stack.multimodal.retrieve(
        query=request.question,
        top_k=request.top_k,
        document_id=request.document_id,
        include_images=request.include_images,
    )
    answer = generator.generate(request.question, evidence)

    conversation = get_or_create_conversation(db, request.conversation_id, request.question)
    record_exchange(db, conversation, request.question, answer, pipeline="fixed")

    return _to_response(request.question, answer, conversation_id=conversation.id)


@router.post("/query/stream")
def ask_stream(
    request: QueryRequest,
    db: Session = Depends(get_db),
    stack=Depends(require_retrieval_stack),
    generator=Depends(require_answer_generator),
    session_factory=Depends(get_session_factory),
) -> StreamingResponse:
    """same pipeline as query, streaming tokens then one result event"""
    evidence = stack.multimodal.retrieve(
        query=request.question,
        top_k=request.top_k,
        document_id=request.document_id,
        include_images=request.include_images,
    )
    try:
        # calling, not iterating, so a missing streaming support fails as a normal 501
        stream = generator.generate_stream(request.question, evidence)
    except NotImplementedError as error:
        raise HTTPException(status_code=501, detail=str(error))

    # committed so a fresh session opened later can see this row
    conversation = get_or_create_conversation(db, request.conversation_id, request.question)
    db.commit()
    conversation_id = conversation.id

    def events():
        answer: Answer | None = None
        for kind, payload in stream:
            if kind == "token":
                yield sse_event("token", {"text": payload})
            else:
                answer = payload

        stream_db = session_factory()
        try:
            conversation = stream_db.get(Conversation, conversation_id)
            record_exchange(stream_db, conversation, request.question, answer, pipeline="fixed")
        finally:
            stream_db.close()

        result = _to_response(request.question, answer, conversation_id=conversation_id)
        yield sse_event("result", result.model_dump())

    return StreamingResponse(events(), media_type="text/event-stream")
