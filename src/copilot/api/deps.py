"""shared fastapi dependencies for retrieval, answering, and the agent"""

import logging

from fastapi import HTTPException

from copilot.agent.deps import AgentUnavailable, get_agent_stack
from copilot.agent.orchestrator import ToolUsingAgent
from copilot.generation.base import AnswerGenerator
from copilot.generation.generator import get_answer_generator
from copilot.retrieval.deps import RetrievalStack, RetrievalUnavailable, get_retrieval_stack

logger = logging.getLogger(__name__)


def get_optional_retrieval_stack() -> RetrievalStack | None:
    """none when retrieval is unavailable, so an upload can still succeed"""
    try:
        return get_retrieval_stack()
    except RetrievalUnavailable:
        return None


def require_retrieval_stack() -> RetrievalStack:
    """503 when retrieval is unavailable, recovering on its own once fixed"""
    try:
        return get_retrieval_stack()
    except RetrievalUnavailable as error:
        raise HTTPException(status_code=503, detail=f"Retrieval unavailable: {error}")


def require_answer_generator() -> AnswerGenerator:
    """503 when the answering model cannot be loaded, kept apart from search"""
    try:
        return get_answer_generator()
    except Exception as error:
        logger.warning("Answer generator unavailable: %s", error)
        raise HTTPException(status_code=503, detail=f"Answer generation unavailable: {error}")


def require_agent() -> ToolUsingAgent:
    """503 when retrieval or the local model cannot be reached"""
    try:
        return get_agent_stack().agent
    except AgentUnavailable as error:
        raise HTTPException(status_code=503, detail=f"Agent unavailable: {error}")
