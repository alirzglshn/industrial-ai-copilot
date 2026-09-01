"""persisting questions and answers as browsable history"""

from sqlalchemy.orm import Session

from copilot.db.models import Conversation, Message
from copilot.generation.base import Answer

TITLE_MAX_LENGTH = 80


def _title_from_question(question: str) -> str:
    question = " ".join(question.split())
    if len(question) <= TITLE_MAX_LENGTH:
        return question
    return question[: TITLE_MAX_LENGTH - 1].rstrip() + "…"


def get_or_create_conversation(
    db: Session, conversation_id: str | None, first_question: str
) -> Conversation:
    if conversation_id:
        conversation = db.get(Conversation, conversation_id)
        if conversation is not None:
            return conversation
        # id given but not found, starting fresh instead of 404
    conversation = Conversation(title=_title_from_question(first_question))
    db.add(conversation)
    db.flush()
    return conversation


def record_exchange(
    db: Session, conversation: Conversation, question: str, answer: Answer, pipeline: str
) -> None:
    db.add(Message(conversation_id=conversation.id, role="user", text=question))
    db.add(
        Message(
            conversation_id=conversation.id,
            role="assistant",
            text=answer.text,
            pipeline=pipeline,
            citations=[
                {
                    "kind": item.kind.value,
                    "document_id": item.document_id,
                    "page_number": item.page_number,
                    "chunk_id": item.chunk_id,
                    "image_id": item.image_id,
                    "image_path": item.image_path,
                }
                for item in answer.evidence_used
            ],
            insufficient_evidence=answer.insufficient_evidence,
            grounded=answer.grounded,
            faithfulness=answer.faithfulness,
            unsupported_pages=answer.unsupported_pages,
            tool_calls=answer.tool_calls,
        )
    )
    db.commit()
