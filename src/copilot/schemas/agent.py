from pydantic import BaseModel, Field


class AgentQueryRequest(BaseModel):
    question: str = Field(min_length=1)
    # restricting every tool call to one manual
    document_id: str | None = None
    # continuing an existing conversation's history, omit to start a new one
    conversation_id: str | None = None

    # no include_images flag, the agent decides for itself whether to call search_images
