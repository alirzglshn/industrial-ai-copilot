"""Phase 6: tool-using agent that decides which retrieval/compute steps a
question needs, instead of always running a fixed text-search-then-answer
pipeline.

Planned tools: search_documents, search_images, get_page, calculate,
get_document_metadata. Implemented in Phase 6, on top of the Phase 3-5
retrieval and generation interfaces. Defined now so the tool contract is
fixed before individual tools are built.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from copilot.generation.base import Answer


@dataclass
class ToolResult:
    tool_name: str
    output: Any


class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult: ...


class Agent(ABC):
    """Decides which tools to call, in what order, to answer a question."""

    @abstractmethod
    def run(self, question: str, document_id: str | None = None) -> Answer: ...
