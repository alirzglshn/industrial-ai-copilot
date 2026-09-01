"""running the evaluation question set against a real corpus and a real model

reuses the same retrieval and generation code paths the api uses, so what
this measures is exactly what an api caller would see

usage:
    python -m eval.run_evaluation --manuals-dir path/to/pdfs
    python -m eval.run_evaluation --manuals-dir path/to/pdfs --use-agent
"""

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from copilot.agent.orchestrator import ToolUsingAgent
from copilot.agent.planner import LlmPlanner
from copilot.agent.tools import (
    CalculatorTool,
    GetDocumentMetadataTool,
    GetPageTool,
    SearchDocumentsTool,
    SearchImagesTool,
)
from copilot.db.models import Base
from copilot.generation.generator import LocalLlmAnswerGenerator
from copilot.generation.local_lm import get_local_lm
from copilot.ingestion.chunker import TextChunker
from copilot.ingestion.parser import PdfDocumentParser, PdfParserConfig
from copilot.ingestion.service import IngestionService
from copilot.retrieval.embedder import SentenceTransformerEmbedder
from copilot.retrieval.image_embedder import ClipImageEmbedder
from copilot.retrieval.image_indexer import ImageIndexer
from copilot.retrieval.image_retriever import DbPageImageSource, ImageRetriever
from copilot.retrieval.indexer import ChunkIndexer
from copilot.retrieval.multimodal import MultimodalRetriever
from copilot.retrieval.retriever import VectorRetriever
from copilot.retrieval.vector_store import QdrantVectorStore

from eval.dataset import EvalQuestion, load_questions
from eval.metrics import QuestionResult, aggregate, precision_at_k, reciprocal_rank, recall_at_k

DEFAULT_QUESTIONS = Path(__file__).parent / "questions.json"
DEFAULT_OUTPUT = Path(__file__).parent / "evaluation_results.json"

TEXT_MODEL = "BAAI/bge-small-en-v1.5"
IMAGE_MODEL = "openai/clip-vit-base-patch32"
ANSWER_MODEL_HUB = "Qwen/Qwen2.5-1.5B-Instruct"
# a local copy skips the hugging face download, same convention as the integration tests
LOCAL_ANSWER_MODEL = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-1.5B-Instruct"


def _answer_model_path() -> str:
    if (LOCAL_ANSWER_MODEL / "model.safetensors").exists():
        return str(LOCAL_ANSWER_MODEL)
    return ANSWER_MODEL_HUB


def build_system(manuals_dir: Path, work_dir: Path, top_k: int):
    engine = create_engine(f"sqlite:///{work_dir / 'eval.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = session_factory()

    print("Loading models (BGE, CLIP, and the answer model)...", file=sys.stderr)
    text_embedder = SentenceTransformerEmbedder(TEXT_MODEL)
    image_embedder = ClipImageEmbedder(IMAGE_MODEL)
    # cached by model name, so LocalLlmAnswerGenerator below reuses this instance
    lm = get_local_lm(_answer_model_path())

    qdrant = QdrantClient(":memory:")
    text_store = QdrantVectorStore(qdrant, "eval_text", text_embedder.dimension)
    text_store.ensure_collection()
    image_store = QdrantVectorStore(qdrant, "eval_images", image_embedder.dimension)
    image_store.ensure_collection()

    service = IngestionService(
        parser=PdfDocumentParser(PdfParserConfig(image_dir=work_dir / "img")),
        chunker=TextChunker(800, 150),
        upload_dir=work_dir / "uploads",
    )
    chunk_indexer = ChunkIndexer(text_embedder, text_store)
    image_indexer = ImageIndexer(image_embedder, image_store)

    pdf_paths = sorted(Path(manuals_dir).glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {manuals_dir}")

    name_to_id: dict[str, str] = {}
    for pdf_path in pdf_paths:
        document = service.ingest(db, pdf_path.name, pdf_path.read_bytes())
        chunk_indexer.index_document(db, document.id)
        image_indexer.index_document(db, document.id)
        name_to_id[pdf_path.stem] = document.id
        print(f"  ingested {pdf_path.name} -> {document.id} ({document.page_count} pages)", file=sys.stderr)

    text_retriever = VectorRetriever(text_embedder, text_store)
    image_retriever = ImageRetriever(image_embedder, image_store)
    multimodal = MultimodalRetriever(
        text_retriever=text_retriever,
        image_retriever=image_retriever,
        page_images=DbPageImageSource(session_factory),
        image_top_k=top_k,
    )

    tools = {
        "search_documents": SearchDocumentsTool(text_retriever, default_top_k=top_k),
        "search_images": SearchImagesTool(image_retriever, default_top_k=top_k),
        "get_page": GetPageTool(session_factory),
        "calculate": CalculatorTool(),
        "get_document_metadata": GetDocumentMetadataTool(session_factory),
    }
    agent = ToolUsingAgent(LlmPlanner(lm, tools), tools, lm)
    fixed_generator = LocalLlmAnswerGenerator(_answer_model_path())

    return multimodal, agent, fixed_generator, name_to_id


def _warn_about_unknown_document_names(questions: list[EvalQuestion], name_to_id: dict[str, str]) -> None:
    referenced = {name for q in questions for name, _ in q.expected_pages}
    unknown = referenced - set(name_to_id)
    if unknown:
        print(f"WARNING: questions reference documents not found in --manuals-dir: {sorted(unknown)}", file=sys.stderr)


def run_question(
    q: EvalQuestion, multimodal, agent, fixed_generator, name_to_id: dict[str, str], top_k: int, use_agent: bool
) -> QuestionResult:
    t0 = time.perf_counter()
    evidence = multimodal.retrieve(q.question, top_k=top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    retrieved_pages = [(e.document_id, e.page_number) for e in evidence]
    expected = {(name_to_id.get(name, name), page) for name, page in q.expected_pages}

    t0 = time.perf_counter()
    answer = agent.run(q.question) if use_agent else fixed_generator.generate(q.question, evidence)
    generation_ms = (time.perf_counter() - t0) * 1000

    expectation_met = None
    if q.category == "refusal":
        expectation_met = answer.insufficient_evidence == q.expect_insufficient
    elif q.category == "calculation":
        expectation_met = any(needle in answer.text for needle in q.answer_must_contain)

    return QuestionResult(
        id=q.id,
        question=q.question,
        category=q.category,
        recall_at_k=recall_at_k(retrieved_pages, expected, top_k),
        precision_at_k=precision_at_k(retrieved_pages, expected, top_k),
        reciprocal_rank=reciprocal_rank(retrieved_pages, expected),
        insufficient_evidence=answer.insufficient_evidence,
        grounded=answer.grounded,
        faithfulness=answer.faithfulness,
        unsupported_pages=answer.unsupported_pages,
        answer_text=answer.text,
        tool_calls=answer.tool_calls,
        expectation_met=expectation_met,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
    )


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "  - "


def print_summary(results: list[QuestionResult], agg) -> None:
    print("\n" + "=" * 82)
    print(f"{'ID':<5} {'category':<12} {'R@K':>6} {'P@K':>6} {'MRR':>6} {'grounded':>9} {'faith':>6} {'ok':>4}")
    for r in results:
        ok = "-" if r.expectation_met is None else ("Y" if r.expectation_met else "N")
        print(
            f"{r.id:<5} {r.category:<12} {_fmt(r.recall_at_k):>6} {_fmt(r.precision_at_k):>6} "
            f"{_fmt(r.reciprocal_rank):>6} {str(r.grounded):>9} {r.faithfulness:>6.2f} {ok:>4}"
        )
    print("=" * 82)
    print(f"Questions:            {agg.n_questions}")
    print(f"Recall@K:             {_fmt(agg.recall_at_k)}")
    print(f"Precision@K:          {_fmt(agg.precision_at_k)}")
    print(f"MRR:                  {_fmt(agg.mrr)}")
    print(f"Expectation accuracy: {_fmt(agg.expectation_accuracy)}  (refusal + calculation questions)")
    print(f"Hallucination rate:   {_fmt(agg.hallucination_rate)}  (of answered, non-refused questions)")
    print(f"Mean faithfulness:    {_fmt(agg.mean_faithfulness)}")
    print(f"Refusal rate:         {agg.refusal_rate:.2f}")
    print(f"Mean retrieval time:  {agg.mean_retrieval_ms:.0f} ms")
    print(f"Mean generation time: {agg.mean_generation_ms:.0f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation harness")
    parser.add_argument("--manuals-dir", type=Path, required=True, help="Directory of PDFs to ingest")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--use-agent", action="store_true", help="Answer via the tool-using agent instead of the fixed pipeline"
    )
    parser.add_argument("--work-dir", type=Path, default=None, help="Where to ingest into (default: ./.eval_work)")
    args = parser.parse_args()

    work_dir = args.work_dir or Path.cwd() / ".eval_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args.questions)
    multimodal, agent, fixed_generator, name_to_id = build_system(args.manuals_dir, work_dir, args.top_k)
    _warn_about_unknown_document_names(questions, name_to_id)

    results = []
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q.question}", file=sys.stderr)
        results.append(
            run_question(q, multimodal, agent, fixed_generator, name_to_id, args.top_k, args.use_agent)
        )

    agg = aggregate(results)
    print_summary(results, agg)

    payload = {
        "generator": "agent" if args.use_agent else "fixed_pipeline",
        "top_k": args.top_k,
        "aggregate": asdict(agg),
        "questions": [asdict(r) for r in results],
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
