from datetime import datetime

from pydantic import BaseModel, ConfigDict

from copilot.schemas.query import Citation


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    text: str
    created_at: datetime
    pipeline: str | None = None
    citations: list[Citation] | None = None
    insufficient_evidence: bool | None = None
    grounded: bool | None = None
    faithfulness: float | None = None
    unsupported_pages: list[int] | None = None
    tool_calls: list[str] | None = None


class ConversationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    message_count: int = 0


class ConversationDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    messages: list[MessageOut]
