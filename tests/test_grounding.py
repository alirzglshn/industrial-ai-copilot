from copilot.generation.grounding import ground
from tests.test_prompt import image_evidence, text_evidence


def test_resolves_a_cited_page_to_its_evidence() -> None:
    evidence = [text_evidence(37), text_evidence(4)]

    result = ground("Caused by low airflow [page 37].", evidence)

    assert [e.page_number for e in result.evidence_used] == [37]
    assert result.unsupported_pages == []
    assert not result.has_unsupported_citations


def test_reports_a_page_that_was_never_in_the_evidence() -> None:
    """The failure this exists to catch: an authoritative-looking invention."""
    result = ground("The limit is 90 C [page 99].", [text_evidence(37)])

    assert result.unsupported_pages == [99]
    assert result.has_unsupported_citations
    assert result.evidence_used == []


def test_separates_real_citations_from_invented_ones() -> None:
    result = ground("Airflow [page 37], and the limit is 90 C [page 99].", [text_evidence(37)])

    assert [e.page_number for e in result.evidence_used] == [37]
    assert result.unsupported_pages == [99]


def test_an_answer_with_no_citations_uses_no_evidence() -> None:
    result = ground("The pump overheats.", [text_evidence(37)])

    assert result.evidence_used == []
    assert result.unsupported_pages == []


def test_citing_a_page_returns_every_piece_of_evidence_on_it() -> None:
    """A page can contribute both a passage and a diagram."""
    evidence = [text_evidence(38), image_evidence(38)]

    result = ground("See [page 38].", evidence)

    assert len(result.evidence_used) == 2
    assert {e.kind for e in result.evidence_used} == {
        evidence[0].kind,
        evidence[1].kind,
    }


def test_evidence_stays_in_retrieval_order() -> None:
    """Most relevant first is what a reader wants, not lowest page first."""
    evidence = [text_evidence(37), text_evidence(4)]

    result = ground("Both [page 4] and [page 37] apply.", evidence)

    assert [e.page_number for e in result.evidence_used] == [37, 4]


def test_unsupported_pages_are_sorted() -> None:
    result = ground("[page 99] [page 12] [page 50]", [])

    assert result.unsupported_pages == [12, 50, 99]


def test_no_evidence_makes_every_citation_unsupported() -> None:
    result = ground("Caused by low airflow [page 37].", [])

    assert result.unsupported_pages == [37]
    assert result.evidence_used == []


def test_duplicate_citations_do_not_duplicate_evidence() -> None:
    result = ground("[page 37] and again [page 37]", [text_evidence(37)])

    assert len(result.evidence_used) == 1
