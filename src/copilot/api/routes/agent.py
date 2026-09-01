"""an agent choosing its own retrieval and computation steps for a question"""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from copilot.api.deps import require_agent
from copilot.api.routes.query import _to_response
from copilot.api.sse import sse_event
from copilot.conversation.service import get_or_create_conversation, record_exchange
from copilot.db.models import Conversation
from copilot.db.session import get_db, get_session_factory
from copilot.generation.base import Answer
from copilot.schemas.agent import AgentQueryRequest
from copilot.schemas.query import QueryResponse

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/query", response_model=QueryResponse)
def ask_agent(
    request: AgentQueryRequest, db: Session = Depends(get_db), agent=Depends(require_agent)
) -> QueryResponse:
    answer = agent.run(request.question, document_id=request.document_id)

    conversation = get_or_create_conversation(db, request.conversation_id, request.question)
    record_exchange(db, conversation, request.question, answer, pipeline="agent")

    return _to_response(request.question, answer, conversation_id=conversation.id)


@router.post("/query/stream")
def ask_agent_stream(
    request: AgentQueryRequest,
    db: Session = Depends(get_db),
    agent=Depends(require_agent),
    session_factory=Depends(get_session_factory),
) -> StreamingResponse:
    """streaming tool_calls, then tokens, then one result event"""
    conversation = get_or_create_conversation(db, request.conversation_id, request.question)
    db.commit()
    conversation_id = conversation.id

    def events():
        answer: Answer | None = None
        for kind, payload in agent.run_stream(request.question, document_id=request.document_id):
            if kind == "tool_calls":
                yield sse_event("tool_calls", {"tool_calls": payload})
            elif kind == "token":
                yield sse_event("token", {"text": payload})
            else:
                answer = payload

        stream_db = session_factory()
        try:
            conversation = stream_db.get(Conversation, conversation_id)
            record_exchange(stream_db, conversation, request.question, answer, pipeline="agent")
        finally:
            stream_db.close()

        result = _to_response(request.question, answer, conversation_id=conversation_id)
        yield sse_event("result", result.model_dump())

    return StreamingResponse(events(), media_type="text/event-stream")
