"""the agent with the real local model, planning and answering for real

deselected by default, downloads weights on first run (or uses the local
copy under models/):
    pytest -m integration

asserts mechanics: the planner always resolves to a valid plan and degrades
to the fallback rather than crashing, not that tool selection is good — the
evaluation harness is where that gets measured
"""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from copilot.agent.orchestrator import ToolUsingAgent
from copilot.agent.planner import LlmPlanner, Plan, ToolCall
from copilot.agent.tools import CalculatorTool, GetDocumentMetadataTool, SearchDocumentsTool
from copilot.generation.local_lm import LocalCausalLM
from copilot.ingestion.chunker import TextChunker
from copilot.ingestion.parser import PdfDocumentParser, PdfParserConfig
from copilot.ingestion.service import IngestionService
from copilot.retrieval.embedder import SentenceTransformerEmbedder
from copilot.retrieval.indexer import ChunkIndexer
from copilot.retrieval.retriever import VectorRetriever
from copilot.retrieval.vector_store import QdrantVectorStore

pytestmark = pytest.mark.integration

HUB_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_COPY = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-1.5B-Instruct"
MODEL = str(LOCAL_COPY) if (LOCAL_COPY / "model.safetensors").exists() else HUB_ID

OVERHEATING = (
    "Overheating is most commonly caused by insufficient cooling airflow "
    "across the motor fins. A blocked intake filter reduces airflow by up to "
    "sixty percent and will trigger the thermal cutout."
)


@pytest.fixture(scope="module")
def real_lm() -> LocalCausalLM:
    return LocalCausalLM(MODEL)


@pytest.fixture
def indexed_manual(db_session, manual_pdf_bytes: bytes, tmp_path: Path):
    """real ingestion and real embeddings, so search_documents has something to find"""
    client = QdrantClient(":memory:")
    embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    store = QdrantVectorStore(client, "agent-integration", embedder.dimension)
    store.ensure_collection()

    service = IngestionService(
        parser=PdfDocumentParser(PdfParserConfig(image_dir=tmp_path / "img")),
        chunker=TextChunker(800, 150),
        upload_dir=tmp_path / "up",
    )
    document = service.ingest(db_session, "manual.pdf", manual_pdf_bytes)
    ChunkIndexer(embedder, store).index_document(db_session, document.id)

    try:
        yield VectorRetriever(embedder, store), document.id
    finally:
        client.close()


def test_real_planner_always_returns_a_valid_plan(real_lm: LocalCausalLM, indexed_manual) -> None:
    """whatever the model's json discipline is, this must never raise or return garbage"""
    retriever, _ = indexed_manual
    tools = {"search_documents": SearchDocumentsTool(retriever), "calculate": CalculatorTool()}
    planner = LlmPlanner(real_lm, tools, max_steps=4)

    plan = planner.plan("Why does the pump overheat?")

    assert isinstance(plan, Plan)
    assert len(plan.calls) <= 4
    assert all(call.tool in tools for call in plan.calls)


def test_real_planner_respects_max_steps_on_a_multi_part_question(
    real_lm: LocalCausalLM, indexed_manual, session_factory
) -> None:
    retriever, _ = indexed_manual
    tools = {
        "search_documents": SearchDocumentsTool(retriever),
        "calculate": CalculatorTool(),
        "get_document_metadata": GetDocumentMetadataTool(session_factory),
    }
    planner = LlmPlanner(real_lm, tools, max_steps=2)

    plan = planner.plan(
        "Compare the cooling specs of every manual, list all documents, "
        "fetch page 5 of each, and calculate the average difference."
    )

    assert len(plan.calls) <= 2


def test_real_agent_answers_a_retrieval_question_without_crashing(
    real_lm: LocalCausalLM, indexed_manual
) -> None:
    retriever, document_id = indexed_manual
    tools = {"search_documents": SearchDocumentsTool(retriever), "calculate": CalculatorTool()}
    planner = LlmPlanner(real_lm, tools, max_steps=3)
    agent = ToolUsingAgent(planner, tools, real_lm, answer_max_new_tokens=150)

    answer = agent.run("What causes the pump to overheat?", document_id=document_id)

    assert isinstance(answer.text, str)
    assert isinstance(answer.tool_calls, list)
    assert isinstance(answer.evidence_used, list)


def test_a_computed_fact_reaches_a_real_generated_answer(real_lm: LocalCausalLM) -> None:
    """bypassing the real planner to isolate whether a computed fact reaches the final answer"""

    class FixedPlanner:
        def plan(self, question: str) -> Plan:
            return Plan(calls=[ToolCall(tool="calculate", arguments={"expression": "(95-80)/80*100"})])

    tools = {"calculate": CalculatorTool()}
    agent = ToolUsingAgent(FixedPlanner(), tools, real_lm, answer_max_new_tokens=100)

    # calculate alone produces a computed fact with no evidence, enough to skip the refusal
    answer = agent.run("What is the percentage difference between 95 and 80?")

    assert not answer.insufficient_evidence
    assert "18.75" in answer.text or "18.8" in answer.text
