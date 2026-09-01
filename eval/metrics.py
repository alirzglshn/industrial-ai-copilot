"""retrieval and generation metrics, as pure functions over plain data"""

from dataclasses import dataclass

# a retrieved or expected item, identified by document_id, page_number
PageKey = tuple[str, int]


def recall_at_k(retrieved: list[PageKey], expected: set[PageKey], k: int) -> float | None:
    """share of the expected pages in the top-k, none rather than 0.0 when nothing was expected"""
    if not expected:
        return None
    top_k = set(retrieved[:k])
    return len(top_k & expected) / len(expected)


def precision_at_k(retrieved: list[PageKey], expected: set[PageKey], k: int) -> float | None:
    """share of the top-k retrieved that are actually relevant"""
    if not expected:
        return None
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in expected)
    return hits / len(top_k)


def reciprocal_rank(retrieved: list[PageKey], expected: set[PageKey]) -> float | None:
    """1/rank of the first relevant hit, 0 if none of the retrieved items are relevant"""
    if not expected:
        return None
    for rank, item in enumerate(retrieved, start=1):
        if item in expected:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float | None:
    """mean of the non-none values, none if there were none to average"""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


@dataclass
class QuestionResult:
    """everything measured for one question, before aggregation"""

    id: str
    question: str
    category: str

    # retrieval, against the top-k the multimodal retriever actually returned
    recall_at_k: float | None
    precision_at_k: float | None
    reciprocal_rank: float | None

    # generation, pulled directly from answer, same fields the api returns
    insufficient_evidence: bool
    grounded: bool
    faithfulness: float
    unsupported_pages: list[int]
    answer_text: str
    tool_calls: list[str]

    # whether the answer matched the question's ground truth, none if no expectation to check
    expectation_met: bool | None

    # system
    retrieval_ms: float
    generation_ms: float


@dataclass
class AggregateMetrics:
    n_questions: int
    recall_at_k: float | None
    precision_at_k: float | None
    mrr: float | None
    # fraction of refusal-control or calculation questions that met their expectation
    expectation_accuracy: float | None
    # of the answered questions, the fraction not grounded or below the faithfulness threshold
    hallucination_rate: float | None
    mean_faithfulness: float | None
    refusal_rate: float
    mean_retrieval_ms: float
    mean_generation_ms: float


def aggregate(results: list[QuestionResult]) -> AggregateMetrics:
    answered = [r for r in results if not r.insufficient_evidence]
    expectations = [r.expectation_met for r in results if r.expectation_met is not None]
    hallucinated = [r for r in answered if not r.grounded]

    return AggregateMetrics(
        n_questions=len(results),
        recall_at_k=mean([r.recall_at_k for r in results if r.recall_at_k is not None]),
        precision_at_k=mean([r.precision_at_k for r in results if r.precision_at_k is not None]),
        mrr=mean([r.reciprocal_rank for r in results if r.reciprocal_rank is not None]),
        expectation_accuracy=(sum(expectations) / len(expectations)) if expectations else None,
        hallucination_rate=(len(hallucinated) / len(answered)) if answered else None,
        mean_faithfulness=mean([r.faithfulness for r in answered]) if answered else None,
        refusal_rate=sum(1 for r in results if r.insufficient_evidence) / len(results) if results else 0.0,
        mean_retrieval_ms=mean([r.retrieval_ms for r in results]) or 0.0,
        mean_generation_ms=mean([r.generation_ms for r in results]) or 0.0,
    )
