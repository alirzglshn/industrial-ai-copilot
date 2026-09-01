"""building the agent, tools, planner and model in one place"""

import logging
from dataclasses import dataclass
from functools import lru_cache

from copilot.agent.orchestrator import ToolUsingAgent
from copilot.agent.planner import LlmPlanner
from copilot.agent.tools import (
    CalculatorTool,
    GetDocumentMetadataTool,
    GetPageTool,
    SearchDocumentsTool,
    SearchImagesTool,
)
from copilot.core.config import Settings, get_settings
from copilot.db.session import SessionLocal
from copilot.generation.local_lm import get_local_lm
from copilot.retrieval.deps import get_retrieval_stack

logger = logging.getLogger(__name__)


class AgentUnavailable(RuntimeError):
    """retrieval or the local model could not be reached"""


@dataclass
class AgentStack:
    agent: ToolUsingAgent


def _build_tools(settings: Settings) -> dict:
    retrieval = get_retrieval_stack()  # already succeeded here
    return {
        "search_documents": SearchDocumentsTool(
            retrieval.retriever, default_top_k=settings.search_top_k
        ),
        "search_images": SearchImagesTool(
            retrieval.image_retriever, default_top_k=settings.image_search_top_k
        ),
        "get_page": GetPageTool(SessionLocal),
        "calculate": CalculatorTool(),
        "get_document_metadata": GetDocumentMetadataTool(SessionLocal),
    }


@lru_cache(maxsize=1)
def get_agent_stack() -> AgentStack:
    settings = get_settings()
    try:
        get_retrieval_stack()  # raises if text search cannot load
        # shared cache key with generation.generator's answerer, one model loaded
        lm = get_local_lm(settings.answer_model)
    except Exception as error:  # retrieval or model failing to load
        logger.warning("Agent unavailable: %s", error)
        raise AgentUnavailable(str(error)) from error

    tools = _build_tools(settings)
    planner = LlmPlanner(
        lm, tools, max_new_tokens=settings.agent_planner_max_new_tokens, max_steps=settings.agent_max_steps
    )
    agent = ToolUsingAgent(
        planner=planner,
        tools=tools,
        lm=lm,
        answer_max_new_tokens=settings.answer_max_new_tokens,
        max_steps=settings.agent_max_steps,
    )
    return AgentStack(agent=agent)
