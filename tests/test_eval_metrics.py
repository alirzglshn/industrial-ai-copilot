from eval.metrics import (
    QuestionResult,
    aggregate,
    mean,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)

DOC = "doc-a"


def pages(*numbers: int) -> list[tuple[str, int]]:
    return [(DOC, n) for n in numbers]


def expected(*numbers: int) -> set[tuple[str, int]]:
    return {(DOC, n) for n in numbers}


# --- recall_at_k ---------------------------------------------------------


def test_recall_perfect_when_all_expected_pages_are_retrieved() -> None:
    assert recall_at_k(pages(1, 2, 3), expected(1, 2), k=3) == 1.0


def test_recall_partial() -> None:
    assert recall_at_k(pages(1, 9, 9), expected(1, 2), k=3) == 0.5


def test_recall_zero_when_none_retrieved() -> None:
    assert recall_at_k(pages(9, 9, 9), expected(1, 2), k=3) == 0.0


def test_recall_only_counts_within_k() -> None:
    """The correct page is retrieved, but ranked below k."""
    assert recall_at_k(pages(9, 9, 1), expected(1), k=2) == 0.0


def test_recall_is_none_when_nothing_expected() -> None:
    """A refusal-control question has no gold pages; this is not a retrieval failure."""
    assert recall_at_k(pages(1, 2), expected(), k=5) is None


# --- precision_at_k --------------------------------------------------------


def test_precision_all_relevant() -> None:
    assert precision_at_k(pages(1, 2), expected(1, 2, 3), k=2) == 1.0


def test_precision_half_relevant() -> None:
    assert precision_at_k(pages(1, 9), expected(1), k=2) == 0.5


def test_precision_with_fewer_results_than_k() -> None:
    """Only one result exists; precision is over what was actually returned, not k."""
    assert precision_at_k(pages(1), expected(1), k=5) == 1.0


def test_precision_zero_results_and_zero_k_slice() -> None:
    assert precision_at_k([], expected(1), k=5) == 0.0


def test_precision_is_none_when_nothing_expected() -> None:
    assert precision_at_k(pages(1), expected(), k=5) is None


# --- reciprocal_rank ---------------------------------------------------------


def test_reciprocal_rank_of_the_first_result() -> None:
    assert reciprocal_rank(pages(1, 2), expected(1)) == 1.0


def test_reciprocal_rank_of_the_second_result() -> None:
    assert reciprocal_rank(pages(9, 1), expected(1)) == 0.5


def test_reciprocal_rank_of_the_third_result() -> None:
    assert reciprocal_rank(pages(9, 9, 1), expected(1)) == 1 / 3


def test_reciprocal_rank_zero_when_never_found() -> None:
    assert reciprocal_rank(pages(9, 9), expected(1)) == 0.0


def test_reciprocal_rank_is_none_when_nothing_expected() -> None:
    assert reciprocal_rank(pages(1), expected()) is None


# --- mean --------------------------------------------------------------------


def test_mean_of_plain_values() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_mean_skips_none() -> None:
    assert mean([1.0, None, 3.0]) == 2.0


def test_mean_of_nothing_is_none() -> None:
    assert mean([]) is None


def test_mean_of_all_none_is_none() -> None:
    assert mean([None, None]) is None


# --- aggregate -----------------------------------------------------------------


def _result(
    id_: str,
    recall: float | None = None,
    grounded: bool = True,
    faithfulness: float = 1.0,
    insufficient: bool = False,
    expectation_met: bool | None = None,
) -> QuestionResult:
    return QuestionResult(
        id=id_,
        question="q",
        category="retrieval",
        recall_at_k=recall,
        precision_at_k=recall,
        reciprocal_rank=recall,
        insufficient_evidence=insufficient,
        grounded=grounded,
        faithfulness=faithfulness,
        unsupported_pages=[],
        answer_text="a",
        tool_calls=[],
        expectation_met=expectation_met,
        retrieval_ms=10.0,
        generation_ms=20.0,
    )


def test_aggregate_averages_retrieval_metrics() -> None:
    agg = aggregate([_result("1", recall=1.0), _result("2", recall=0.0)])

    assert agg.recall_at_k == 0.5
    assert agg.n_questions == 2


def test_aggregate_hallucination_rate_only_counts_answered_questions() -> None:
    """A refusal is not a hallucination; it made no claim to be wrong about."""
    results = [
        _result("1", grounded=False, insufficient=False),  # answered, ungrounded
        _result("2", grounded=True, insufficient=False),  # answered, grounded
        _result("3", grounded=False, insufficient=True),  # refused: excluded
    ]

    agg = aggregate(results)

    assert agg.hallucination_rate == 0.5  # 1 of 2 *answered* questions
    assert agg.refusal_rate == 1 / 3


def test_aggregate_expectation_accuracy_ignores_questions_with_no_expectation() -> None:
    results = [
        _result("1", expectation_met=True),
        _result("2", expectation_met=False),
        _result("3", expectation_met=None),
    ]

    assert aggregate(results).expectation_accuracy == 0.5


def test_aggregate_of_no_questions_does_not_crash() -> None:
    agg = aggregate([])

    assert agg.n_questions == 0
    assert agg.recall_at_k is None
    assert agg.hallucination_rate is None
    assert agg.refusal_rate == 0.0
