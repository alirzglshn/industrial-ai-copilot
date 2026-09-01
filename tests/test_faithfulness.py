"""Does the cited evidence actually support the answer?

The case that motivated this: a 1.5B model answered "What is the capital city
of France?" with "The capital city of France is Paris [page 21]" against real
pump manuals. Page 21 was genuinely retrieved, so citation resolution passed
it. The claim came entirely from pretraining.
"""

from copilot.generation.faithfulness import measure
from tests.test_prompt import image_evidence, text_evidence

AIRFLOW = (
    "Overheating is most commonly caused by insufficient cooling airflow "
    "across the motor fins."
)


def test_the_citation_marker_itself_is_not_scored_as_unsupported_content() -> None:
    """"[page 37]" must not be graded as a claim needing support.

    "page" is not a stopword, so a correctly-cited answer would otherwise be
    penalized for the act of citing.
    """
    result = measure("Insufficient airflow causes it [page 37].", [text_evidence(37, AIRFLOW)])

    assert "page" not in result.unsupported_terms
    assert "37" not in result.unsupported_terms
    assert result.score == 1.0


def test_an_answer_copied_from_the_evidence_scores_high() -> None:
    result = measure(
        "Overheating is caused by insufficient cooling airflow across the motor fins.",
        [text_evidence(37, AIRFLOW)],
    )

    assert result.score > 0.9
    assert result.unsupported_terms == []


def test_a_pretraining_answer_wearing_a_real_citation_scores_low() -> None:
    """The exact failure citation resolution alone waves through."""
    result = measure(
        "The capital city of France is Paris.", [text_evidence(21, AIRFLOW)]
    )

    assert result.score < 0.3
    assert "paris" in result.unsupported_terms
    assert "france" in result.unsupported_terms


def test_function_words_do_not_inflate_the_score() -> None:
    """An unrelated answer shares 'the' and 'is' with any page whatsoever."""
    result = measure("The pump is the thing that is in the manual.", [text_evidence(1, AIRFLOW)])

    assert result.score < 0.5


def test_paraphrase_using_the_manuals_vocabulary_still_scores_well() -> None:
    result = measure(
        "Insufficient airflow across the fins causes overheating.",
        [text_evidence(37, AIRFLOW)],
    )

    assert result.score > 0.9


def test_evidence_from_several_pages_is_pooled() -> None:
    result = measure(
        "Overheating is caused by insufficient airflow. Replace the intake filter.",
        [
            text_evidence(37, AIRFLOW),
            text_evidence(38, "Replace the intake filter every 500 hours."),
        ],
    )

    assert result.score == 1.0


def test_citing_only_an_uncaptioned_diagram_cannot_support_a_claim() -> None:
    """There is no text there to have said anything."""
    result = measure("The limit is 90 degrees.", [image_evidence(38)])

    assert result.score == 0.0


def test_a_captioned_diagram_can_support_a_claim() -> None:
    result = measure(
        "The impeller clearance is shown.", [image_evidence(38, "impeller clearance diagram")]
    )

    assert result.score > 0.5


def test_citing_nothing_supports_nothing() -> None:
    assert measure("The pump overheats.", []).score == 0.0


def test_an_answer_with_no_content_words_claims_nothing() -> None:
    result = measure("It is what it is.", [text_evidence(1, AIRFLOW)])

    assert result.score == 1.0
    assert result.unsupported_terms == []


def test_numbers_are_checked_like_any_other_term() -> None:
    """A fabricated specification is exactly what must not slip through."""
    result = measure(
        "The maximum temperature is 250 degrees.",
        [text_evidence(12, "The maximum temperature is 80 degrees.")],
    )

    assert "250" in result.unsupported_terms
