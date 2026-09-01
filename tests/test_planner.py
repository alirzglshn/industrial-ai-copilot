import json

import pytest

from copilot.agent.planner import (
    LlmPlanner,
    PlanParseError,
    ToolCall,
    _extract_json_array,
    build_planner_prompt,
    default_fallback_plan,
    parse_plan,
)
from copilot.agent.tools import (
    CalculatorTool,
    GetDocumentMetadataTool,
    SearchDocumentsTool,
    SearchImagesTool,
)
from tests.fakes import ScriptedLocalLM

KNOWN = {"search_documents", "search_images", "calculate", "get_page", "get_document_metadata"}


# --- _extract_json_array -----------------------------------------------------


def test_extracts_a_clean_array() -> None:
    assert _extract_json_array('[{"tool": "calculate", "arguments": {}}]') == [
        {"tool": "calculate", "arguments": {}}
    ]


def test_extracts_an_array_prefaced_by_stray_words() -> None:
    """small models routinely ignore "output only json" and add a lead-in"""
    raw = 'Sure, here is the plan: [{"tool": "calculate", "arguments": {}}]'
    assert _extract_json_array(raw) == [{"tool": "calculate", "arguments": {}}]


def test_extracts_an_array_followed_by_trailing_text() -> None:
    raw = '[{"tool": "calculate", "arguments": {}}] Let me know if you need more.'
    assert _extract_json_array(raw) == [{"tool": "calculate", "arguments": {}}]


def test_no_brackets_at_all_raises() -> None:
    with pytest.raises(PlanParseError):
        _extract_json_array("I think we should search the documents.")


def test_malformed_json_raises() -> None:
    with pytest.raises(PlanParseError):
        _extract_json_array('[{"tool": "calculate", "arguments": }]')


def test_a_json_object_instead_of_an_array_raises() -> None:
    with pytest.raises(PlanParseError):
        _extract_json_array('{"tool": "calculate"}')


def test_an_empty_array_is_valid() -> None:
    assert _extract_json_array("[]") == []


# --- parse_plan ---------------------------------------------------------------


def test_parses_a_valid_plan() -> None:
    raw = json.dumps(
        [
            {"tool": "search_documents", "arguments": {"query": "cooling"}},
            {"tool": "calculate", "arguments": {"expression": "1+1"}},
        ]
    )

    plan = parse_plan(raw, known_tools=KNOWN)

    assert plan.calls == [
        ToolCall(tool="search_documents", arguments={"query": "cooling"}),
        ToolCall(tool="calculate", arguments={"expression": "1+1"}),
    ]
    assert plan.used_fallback is False


def test_an_intentional_empty_plan_is_respected() -> None:
    """the model deciding no tools are needed is a valid answer, not a failure"""
    plan = parse_plan("[]", known_tools=KNOWN)

    assert plan.calls == []
    assert plan.used_fallback is False


def test_items_naming_an_unknown_tool_are_dropped() -> None:
    raw = json.dumps(
        [
            {"tool": "search_documents", "arguments": {"query": "cooling"}},
            {"tool": "delete_everything", "arguments": {}},
        ]
    )

    plan = parse_plan(raw, known_tools=KNOWN)

    assert [call.tool for call in plan.calls] == ["search_documents"]


def test_items_with_non_dict_arguments_are_dropped() -> None:
    raw = json.dumps(
        [
            {"tool": "search_documents", "arguments": "cooling"},
            {"tool": "calculate", "arguments": {"expression": "1+1"}},
        ]
    )

    plan = parse_plan(raw, known_tools=KNOWN)

    assert [call.tool for call in plan.calls] == ["calculate"]


def test_non_object_items_are_dropped() -> None:
    raw = json.dumps(["search_documents", {"tool": "calculate", "arguments": {"expression": "1"}}])

    plan = parse_plan(raw, known_tools=KNOWN)

    assert [call.tool for call in plan.calls] == ["calculate"]


def test_an_array_where_every_item_is_invalid_raises() -> None:
    """every item invalid is the same failure mode as unparseable output"""
    raw = json.dumps([{"tool": "delete_everything", "arguments": {}}])

    with pytest.raises(PlanParseError):
        parse_plan(raw, known_tools=KNOWN)


def test_missing_arguments_key_defaults_to_empty_dict() -> None:
    raw = json.dumps([{"tool": "get_document_metadata"}])

    plan = parse_plan(raw, known_tools=KNOWN)

    assert plan.calls == [ToolCall(tool="get_document_metadata", arguments={})]


# --- default_fallback_plan -----------------------------------------------------


def test_fallback_plan_matches_the_default_search_behaviour() -> None:
    plan = default_fallback_plan("why does it overheat?")

    assert plan.used_fallback is True
    assert [call.tool for call in plan.calls] == ["search_documents", "search_images"]
    assert plan.calls[0].arguments == {"query": "why does it overheat?"}


def test_fallback_plan_without_images() -> None:
    plan = default_fallback_plan("why does it overheat?", include_images=False)

    assert [call.tool for call in plan.calls] == ["search_documents"]


# --- build_planner_prompt -------------------------------------------------------


def test_prompt_lists_every_tool() -> None:
    tools = {"calculate": CalculatorTool()}
    system, _ = build_planner_prompt("q", tools, documents=[], max_steps=4)

    assert "calculate" in system
    assert CalculatorTool.description in system


def test_prompt_includes_available_documents() -> None:
    documents = [{"id": "doc-a", "filename": "manual.pdf", "page_count": 22}]
    system, _ = build_planner_prompt("q", {}, documents=documents, max_steps=4)

    assert "manual.pdf" in system
    assert "doc-a" in system


def test_prompt_states_no_documents_when_none_uploaded() -> None:
    system, _ = build_planner_prompt("q", {}, documents=[], max_steps=4)

    assert "none uploaded" in system


def test_prompt_carries_the_question() -> None:
    _, user = build_planner_prompt("why does the pump overheat?", {}, documents=[], max_steps=4)

    assert "why does the pump overheat?" in user


# --- LlmPlanner ----------------------------------------------------------------


def test_planner_parses_a_valid_model_plan() -> None:
    raw = json.dumps([{"tool": "calculate", "arguments": {"expression": "1+1"}}])
    lm = ScriptedLocalLM([raw])
    planner = LlmPlanner(lm, tools={"calculate": CalculatorTool()})

    plan = planner.plan("what is 1+1?")

    assert plan.used_fallback is False
    assert plan.calls == [ToolCall(tool="calculate", arguments={"expression": "1+1"})]


def test_planner_falls_back_when_the_model_output_is_unparseable(
    retrieval_stack,
) -> None:
    lm = ScriptedLocalLM(["I would search the documents for this."])
    planner = LlmPlanner(
        lm,
        tools={
            "search_documents": SearchDocumentsTool(retrieval_stack.retriever),
            "search_images": SearchImagesTool(retrieval_stack.image_retriever),
        },
    )

    plan = planner.plan("why does it overheat?")

    assert plan.used_fallback is True
    assert [call.tool for call in plan.calls] == ["search_documents", "search_images"]


def test_planner_truncates_a_plan_longer_than_max_steps() -> None:
    raw = json.dumps([{"tool": "calculate", "arguments": {"expression": str(i)}} for i in range(10)])
    lm = ScriptedLocalLM([raw])
    planner = LlmPlanner(lm, tools={"calculate": CalculatorTool()}, max_steps=2)

    plan = planner.plan("do lots of math")

    assert len(plan.calls) == 2


def test_planner_injects_document_context_via_the_metadata_tool(
    db_session, session_factory
) -> None:
    from copilot.db.models import Document

    db_session.add(Document(id="doc-a", filename="grundfos_ups3.pdf", status="indexed", page_count=22))
    db_session.commit()

    lm = ScriptedLocalLM(["[]"])
    planner = LlmPlanner(
        lm,
        tools={
            "calculate": CalculatorTool(),
            "get_document_metadata": GetDocumentMetadataTool(session_factory),
        },
    )

    planner.plan("how many pages does grundfos_ups3 have?")

    system_prompt_sent = lm.calls[0][0]
    assert "grundfos_ups3.pdf" in system_prompt_sent


def test_planner_without_a_metadata_tool_still_plans(db_session, session_factory) -> None:
    """document context is a nice-to-have, not a hard dependency"""
    lm = ScriptedLocalLM(["[]"])
    planner = LlmPlanner(lm, tools={"calculate": CalculatorTool()})

    plan = planner.plan("what is 2+2?")

    assert plan.calls == []
    assert plan.used_fallback is False
