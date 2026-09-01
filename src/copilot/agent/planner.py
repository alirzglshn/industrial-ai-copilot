"""single-shot planning of which tools a question needs, with a fallback plan"""

import json
import logging
from dataclasses import dataclass, field

from copilot.agent.base import Tool
from copilot.generation.local_lm import LocalCausalLM

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 4


@dataclass
class ToolCall:
    tool: str
    arguments: dict


@dataclass
class Plan:
    calls: list[ToolCall] = field(default_factory=list)
    # true when this is the safety-net default, not a model-produced plan
    used_fallback: bool = False


class PlanParseError(ValueError):
    pass


def default_fallback_plan(question: str, include_images: bool = True) -> Plan:
    """search then answer, same as the fixed pipeline"""
    calls = [ToolCall(tool="search_documents", arguments={"query": question})]
    if include_images:
        calls.append(ToolCall(tool="search_images", arguments={"query": question}))
    return Plan(calls=calls, used_fallback=True)


def _extract_json_array(raw: str) -> list:
    """finding the first json array in free-form model output"""
    start = raw.find("[")
    if start == -1:
        raise PlanParseError("no JSON array found in planner output")

    try:
        value, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError as error:
        raise PlanParseError(f"invalid JSON: {error}") from error

    if not isinstance(value, list):
        raise PlanParseError("planner output was not a JSON array")
    return value


def _validate_items(items: list, known_tools: set[str]) -> list[ToolCall]:
    calls = []
    for item in items:
        if not isinstance(item, dict):
            continue
        tool = item.get("tool")
        arguments = item.get("arguments", {})
        if not isinstance(tool, str) or tool not in known_tools:
            continue
        if not isinstance(arguments, dict):
            continue
        calls.append(ToolCall(tool=tool, arguments=arguments))
    return calls


def parse_plan(raw: str, known_tools: set[str]) -> Plan:
    """parsing and validating a planner completion into a plan"""
    items = _extract_json_array(raw)
    calls = _validate_items(items, known_tools)
    if items and not calls:
        raise PlanParseError("no valid tool calls found among parsed items")
    return Plan(calls=calls, used_fallback=False)


def _describe_tool(tool: Tool) -> str:
    params = ", ".join(f"{name} ({desc})" for name, desc in tool.parameters.items())
    return f"- {tool.name}({params}): {tool.description}"


PLANNER_SYSTEM_PROMPT = """You are a planning assistant for a technical-manual question-answering system.

Given a question, decide which tools are needed to answer it, in order. \
Output ONLY a JSON array of steps and nothing else — no explanation, no markdown.

Each step has this exact shape: {{"tool": "<tool name>", "arguments": {{...}}}}

Available tools:
{tool_list}

Rules:
- Use search_documents for questions about what a manual says (how, why, what, when).
- Use search_images only when the question is specifically about a diagram, picture, or visual.
- Use calculate for any arithmetic — differences, percentages, totals. Never compute \
arithmetic yourself in the final answer without a calculate step; the final answer step \
cannot do arithmetic reliably, so unsupported numbers in it will be flagged as invented.
- Use get_page when the question names an exact page number.
- Use get_document_metadata for questions about which manuals exist, or how many pages one has.
- When a question is about a specific manual by name, pass its id as document_id, using the \
list below.
- Output at most {max_steps} steps.
- If a question can be answered with no tools at all, output [].

Available documents:
{document_list}

Worked example — a question needing search plus a calculation:
QUESTION: Model A reaches 95C and model B's limit is 80C. What is the percentage difference?
PLAN: [{{"tool": "search_documents", "arguments": {{"query": "temperature limit specification"}}}}, \
{{"tool": "calculate", "arguments": {{"expression": "(95-80)/80*100"}}}}]

Worked example — a question needing no tools:
QUESTION: What is 12 times 7?
PLAN: [{{"tool": "calculate", "arguments": {{"expression": "12*7"}}}}]
"""


def build_planner_prompt(question: str, tools: dict[str, Tool], documents: list[dict], max_steps: int) -> tuple[str, str]:
    tool_list = "\n".join(_describe_tool(tool) for tool in tools.values())
    document_list = json.dumps(documents) if documents else "(none uploaded yet)"
    system = PLANNER_SYSTEM_PROMPT.format(
        tool_list=tool_list, document_list=document_list, max_steps=max_steps
    )
    user = f"QUESTION: {question.strip()}\nPLAN:"
    return system, user


class LlmPlanner:
    def __init__(
        self,
        lm: LocalCausalLM,
        tools: dict[str, Tool],
        max_new_tokens: int = 220,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self.lm = lm
        self.tools = tools
        self.max_new_tokens = max_new_tokens
        self.max_steps = max_steps

    def _document_context(self) -> list[dict]:
        """manuals available right now, for scoping searches across documents"""
        metadata_tool = self.tools.get("get_document_metadata")
        if metadata_tool is None:
            return []
        try:
            result = metadata_tool.run()
        except Exception:
            logger.warning("Could not load document context for planning", exc_info=True)
            return []
        return [
            {"id": row["id"], "filename": row["filename"], "page_count": row["page_count"]}
            for row in result.output
        ]

    def plan(self, question: str) -> Plan:
        documents = self._document_context()
        system, user = build_planner_prompt(question, self.tools, documents, self.max_steps)
        raw = self.lm.chat(system, user, self.max_new_tokens)

        try:
            plan = parse_plan(raw, known_tools=set(self.tools))
        except PlanParseError as error:
            logger.warning(
                "Planner output for %r could not be parsed (%s); falling back to default search",
                question,
                error,
            )
            return default_fallback_plan(question, include_images="search_images" in self.tools)

        return Plan(calls=plan.calls[: self.max_steps], used_fallback=False)
