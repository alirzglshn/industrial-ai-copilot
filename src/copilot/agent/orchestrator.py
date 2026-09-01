"""running a plan and turning the results into a grounded answer"""

import logging
from collections.abc import Iterator
from dataclasses import replace

from copilot.agent.base import Agent, Tool, ToolResult
from copilot.agent.planner import LlmPlanner, ToolCall
from copilot.generation.base import Answer
from copilot.generation.generator import _finish, _no_evidence_answer, stream_answer
from copilot.generation.local_lm import LocalCausalLM
from copilot.generation.prompt import build_prompt
from copilot.retrieval.base import Evidence, EvidenceKind

logger = logging.getLogger(__name__)

# tools that return citable evidence vs a plain trusted fact
_EVIDENCE_TOOLS = frozenset({"search_documents", "search_images", "get_page"})


def _dedupe_evidence(evidence: list[Evidence]) -> list[Evidence]:
    """dropping duplicate passages or diagrams found by more than one call"""
    seen: set[tuple] = set()
    deduped: list[Evidence] = []
    for item in evidence:
        key = ("image", item.image_id) if item.kind is EvidenceKind.IMAGE else ("text", item.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _format_metadata_fact(row: dict) -> str:
    return (
        f"Document '{row['filename']}' (id {row['id']}): {row['page_count']} pages, "
        f"status {row['status']}."
    )


def _summarize_call(call: ToolCall, result: ToolResult | None, error: str | None) -> str:
    args = ", ".join(f"{key}={value!r}" for key, value in call.arguments.items())
    if error is not None:
        return f"{call.tool}({args}) -> error: {error}"
    if isinstance(result.output, list):
        return f"{call.tool}({args}) -> {len(result.output)} result(s)"
    return f"{call.tool}({args}) -> {result.output!r}"


class ToolUsingAgent(Agent):
    def __init__(
        self,
        planner: LlmPlanner,
        tools: dict[str, Tool],
        lm: LocalCausalLM,
        answer_max_new_tokens: int = 300,
        max_steps: int = 4,
    ) -> None:
        self.planner = planner
        self.tools = tools
        self.lm = lm
        self.answer_max_new_tokens = answer_max_new_tokens
        self.max_steps = max_steps

    def _run_call(self, call: ToolCall, document_id: str | None) -> tuple[ToolResult | None, str | None]:
        tool = self.tools.get(call.tool)
        if tool is None:
            # should not happen, the planner already validates tool names
            return None, f"unknown tool {call.tool!r}"

        arguments = dict(call.arguments)
        if document_id and "document_id" not in arguments and call.tool in _EVIDENCE_TOOLS:
            # scoping to one manual, unless the plan already scoped elsewhere
            arguments["document_id"] = document_id

        try:
            return tool.run(**arguments), None
        except TypeError as error:
            # missing or extra arguments the planner's json check cannot catch
            return None, f"bad arguments: {error}"
        except Exception as error:
            return None, str(error)

    def _gather(
        self, question: str, document_id: str | None
    ) -> tuple[list[Evidence], list[str], list[str]]:
        """planning, running every call, returning (evidence, computed_facts, trace)"""
        plan = self.planner.plan(question)

        evidence: list[Evidence] = []
        computed_facts: list[str] = []
        trace: list[str] = []

        for call in plan.calls[: self.max_steps]:
            result, error = self._run_call(call, document_id)
            trace.append(_summarize_call(call, result, error))

            if error is not None:
                logger.warning("Tool call %s failed for %r: %s", call.tool, question, error)
                continue

            if call.tool in _EVIDENCE_TOOLS:
                evidence.extend(result.output)
            elif call.tool == "calculate":
                computed_facts.append(f"{call.arguments.get('expression')} = {result.output}")
            elif call.tool == "get_document_metadata":
                computed_facts.extend(_format_metadata_fact(row) for row in result.output)

        return _dedupe_evidence(evidence), computed_facts, trace

    def run(self, question: str, document_id: str | None = None) -> Answer:
        evidence, computed_facts, trace = self._gather(question, document_id)

        if not evidence and not computed_facts:
            answer = _no_evidence_answer()
            return replace(answer, tool_calls=trace)

        prompt = build_prompt(question, evidence, computed_facts=computed_facts or None)
        raw = self.lm.chat(prompt.system, prompt.user, self.answer_max_new_tokens)
        answer = _finish(question, raw, evidence)
        return replace(answer, tool_calls=trace)

    def run_stream(
        self, question: str, document_id: str | None = None
    ) -> Iterator[tuple[str, object]]:
        """same flow as run(), yielding tool_calls, then tokens, then done"""
        evidence, computed_facts, trace = self._gather(question, document_id)
        yield ("tool_calls", trace)

        for kind, payload in stream_answer(
            self.lm, question, evidence, self.answer_max_new_tokens, computed_facts=computed_facts or None
        ):
            if kind == "done":
                yield ("done", replace(payload, tool_calls=trace))
            else:
                yield (kind, payload)
