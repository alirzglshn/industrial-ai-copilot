"""Shared FastAPI dependencies for the retrieval stack.

Two flavours, because the two callers want different failure behaviour:
indexing on upload is best-effort, while searching without a retriever is a
service failure the caller must see.
"""

from fastapi import HTTPException

from copilot.retrieval.deps import RetrievalStack, RetrievalUnavailable, get_retrieval_stack


def get_optional_retrieval_stack() -> RetrievalStack | None:
    """None when retrieval is unavailable, so an upload can still succeed."""
    try:
        return get_retrieval_stack()
    except RetrievalUnavailable:
        return None


def require_retrieval_stack() -> RetrievalStack:
    """503 when retrieval is unavailable.

    Failures are not cached, so the endpoint recovers by itself once the
    embedding model or Qdrant becomes reachable.
    """
    try:
        return get_retrieval_stack()
    except RetrievalUnavailable as error:
        raise HTTPException(status_code=503, detail=f"Retrieval unavailable: {error}")
