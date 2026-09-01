"""agent and tool contracts"""

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
    # parameter name to description, shown to the planner
    parameters: dict[str, str] = {}

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult: ...


class Agent(ABC):
    """choosing which tools to call, in what order, to answer a question"""

    @abstractmethod
    def run(self, question: str, document_id: str | None = None) -> Answer: ...
