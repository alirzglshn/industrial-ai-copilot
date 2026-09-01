"""The parts of answering that decide what an answer *means*.

Model loading is not involved: these cover refusal detection, the empty-output
guard, the no-evidence short circuit, and how grounding results become an
Answer — the logic that determines whether the system claims something.
"""

from copilot.generation.generator import _finish, _is_refusal, _no_evidence_answer
from copilot.generation.prompt import INSUFFICIENT_MARKER
from tests.test_prompt import text_evidence


# --- refusal detection ----------------------------------------------------


def test_bare_sentinel_is_a_refusal() -> None:
    assert _is_refusal(INSUFFICIENT_MARKER)


def test_sentinel_wrapped_in_prose_is_still_a_refusal() -> None:
    """Small models routinely pad the sentinel despite being told not to.

    Treating that as a normal answer would convert a correct refusal into an
    unsupported claim.
    """
    assert _is_refusal(f"I'm sorry, but {INSUFFICIENT_MARKER} to answer this.")


def test_refusal_detection_ignores_case() -> None:
    assert _is_refusal("insufficient_evidence")


def test_an_ordinary_answer_is_not_a_refusal() -> None:
    assert not _is_refusal("Overheating is caused by low airflow [page 37].")


# --- finishing an answer --------------------------------------------------


def test_grounded_answer_reports_the_evidence_it_cited() -> None:
    evidence = [text_evidence(37), text_evidence(4)]

    answer = _finish("q", "Low airflow [page 37].", evidence)

    assert not answer.insufficient_evidence
    assert [e.page_number for e in answer.evidence_used] == [37]
    assert answer.unsupported_pages == []


def test_invented_citation_is_surfaced_on_the_answer() -> None:
    answer = _finish("q", "The limit is 90 C [page 99].", [text_evidence(37)])

    assert answer.unsupported_pages == [99]
    # The answer is still returned; the caller decides what to do about it.
    assert not answer.insufficient_evidence


def test_refusal_carries_no_citations() -> None:
    answer = _finish("q", INSUFFICIENT_MARKER, [text_evidence(37)])

    assert answer.insufficient_evidence
    assert answer.evidence_used == []


def test_empty_generation_is_treated_as_insufficient() -> None:
    """A blank must not read as a confident empty answer."""
    answer = _finish("q", "   \n  ", [text_evidence(37)])

    assert answer.insufficient_evidence
    assert answer.text == ""
    assert answer.evidence_used == []


def test_answer_text_is_stripped() -> None:
    answer = _finish("q", "  Low airflow [page 37].\n ", [text_evidence(37)])

    assert answer.text == "Low airflow [page 37]."


def test_uncited_answer_reports_no_evidence_used() -> None:
    """Not a refusal, but nothing backs it either, which is worth seeing."""
    answer = _finish("q", "The pump overheats.", [text_evidence(37)])

    assert not answer.insufficient_evidence
    assert answer.evidence_used == []


# --- groundedness ---------------------------------------------------------


def test_a_cited_answer_is_grounded() -> None:
    assert _finish("q", "Low airflow [page 37].", [text_evidence(37)]).grounded


def test_an_answer_citing_nothing_is_not_grounded() -> None:
    """The failure that let 'France has no capital city' through as a clean answer.

    The text may even be correct, but nothing in the manual backs it, so it
    must not be presented as sourced.
    """
    assert not _finish("q", "The pump overheats.", [text_evidence(37)]).grounded


def test_an_answer_citing_only_invented_pages_is_not_grounded() -> None:
    answer = _finish("q", "The limit is 90 C [page 99].", [text_evidence(37)])

    assert not answer.grounded
    assert answer.unsupported_pages == [99]


def test_a_refusal_counts_as_grounded() -> None:
    """A refusal makes no claim, so there is nothing to support."""
    assert _finish("q", INSUFFICIENT_MARKER, [text_evidence(37)]).grounded


def test_no_evidence_refusal_counts_as_grounded() -> None:
    assert _no_evidence_answer().grounded


# --- no evidence ----------------------------------------------------------


def test_no_evidence_short_circuits_to_a_refusal() -> None:
    """Calling a model with an empty evidence block invites it to answer from
    pretraining, which is the exact failure this system exists to avoid."""
    answer = _no_evidence_answer()

    assert answer.insufficient_evidence
    assert answer.evidence_used == []
    assert INSUFFICIENT_MARKER in answer.text
