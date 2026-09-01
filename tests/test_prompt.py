from copilot.generation.prompt import (
    INSUFFICIENT_MARKER,
    build_prompt,
    cited_pages,
)
from copilot.retrieval.base import Evidence, EvidenceKind


def text_evidence(page: int, body: str = "cooling airflow matters") -> Evidence:
    return Evidence(
        kind=EvidenceKind.TEXT,
        document_id="doc-a",
        page_number=page,
        score=0.7,
        chunk_id=f"c{page}",
        text=body,
    )


def image_evidence(page: int, caption: str | None = None) -> Evidence:
    return Evidence(
        kind=EvidenceKind.IMAGE,
        document_id="doc-a",
        page_number=page,
        score=0.3,
        image_id=f"i{page}",
        image_path=f"/images/p{page}.png",
        text=caption,
    )


def test_system_prompt_states_the_four_rules() -> None:
    prompt = build_prompt("why does it overheat?", [text_evidence(37)])

    lowered = prompt.system.lower()
    assert "only from the evidence" in lowered
    assert INSUFFICIENT_MARKER in prompt.system
    assert "[page 37]" in prompt.system or "page 37" in prompt.system
    assert "inference" in lowered


def test_evidence_is_numbered_and_carries_page_numbers() -> None:
    prompt = build_prompt("q", [text_evidence(37), text_evidence(38)])

    assert "[1] (page 37)" in prompt.user
    assert "[2] (page 38)" in prompt.user


def test_question_is_included() -> None:
    prompt = build_prompt("why does the pump overheat?", [text_evidence(1)])

    assert "why does the pump overheat?" in prompt.user


def test_images_are_announced_with_their_page() -> None:
    """Without this the model cannot point a reader at a figure."""
    prompt = build_prompt("q", [image_evidence(38)])

    assert "(page 38)" in prompt.user
    assert "Diagram" in prompt.user


def test_image_caption_is_used_when_present() -> None:
    prompt = build_prompt("q", [image_evidence(38, caption="impeller clearance")])

    assert "Diagram: impeller clearance" in prompt.user


def test_uncaptioned_image_says_so_rather_than_looking_blank() -> None:
    prompt = build_prompt("q", [image_evidence(38)])

    assert "no caption available" in prompt.user


def test_image_paths_are_collected_for_a_vlm() -> None:
    prompt = build_prompt("q", [text_evidence(1), image_evidence(38), image_evidence(39)])

    assert prompt.image_paths == ["/images/p38.png", "/images/p39.png"]


def test_text_evidence_contributes_no_image_paths() -> None:
    assert build_prompt("q", [text_evidence(1)]).image_paths == []


def test_whitespace_in_evidence_is_normalized() -> None:
    prompt = build_prompt("q", [text_evidence(1, "line one\n\n   line   two")])

    assert "line one line two" in prompt.user


def test_computed_facts_appear_in_their_own_section() -> None:
    prompt = build_prompt("q", [text_evidence(1)], computed_facts=["(95-80)/80*100 = 18.75"])

    assert "COMPUTED FACTS" in prompt.user
    assert "(95-80)/80*100 = 18.75" in prompt.user
    assert "no page citation needed" in prompt.user


def test_no_computed_facts_produces_no_facts_section() -> None:
    prompt = build_prompt("q", [text_evidence(1)])

    assert "COMPUTED FACTS" not in prompt.user


def test_empty_computed_facts_list_produces_no_facts_section() -> None:
    prompt = build_prompt("q", [text_evidence(1)], computed_facts=[])

    assert "COMPUTED FACTS" not in prompt.user


def test_computed_facts_come_after_evidence_and_before_the_question() -> None:
    prompt = build_prompt("the actual question", [text_evidence(1)], computed_facts=["2 + 2 = 4"])

    evidence_pos = prompt.user.index("EVIDENCE:")
    facts_pos = prompt.user.index("COMPUTED FACTS")
    question_pos = prompt.user.index("QUESTION:")
    assert evidence_pos < facts_pos < question_pos


def test_multiple_computed_facts_are_each_on_their_own_line() -> None:
    prompt = build_prompt("q", [text_evidence(1)], computed_facts=["fact one", "fact two"])

    assert "- fact one" in prompt.user
    assert "- fact two" in prompt.user


def test_empty_evidence_is_stated_explicitly() -> None:
    prompt = build_prompt("q", [])

    assert "(no evidence retrieved)" in prompt.user


# --- citation parsing -----------------------------------------------------


def test_finds_cited_pages() -> None:
    assert cited_pages("Caused by low airflow [page 37] and see [page 38].") == {37, 38}


def test_citation_parsing_is_case_insensitive() -> None:
    assert cited_pages("see [Page 12]") == {12}


def test_tolerates_extra_spacing() -> None:
    assert cited_pages("see [page   9]") == {9}


def test_repeated_citations_collapse() -> None:
    assert cited_pages("[page 5] and again [page 5]") == {5}


def test_no_citations_returns_empty() -> None:
    assert cited_pages("The pump overheats.") == set()


def test_bare_page_mentions_are_not_citations() -> None:
    """Only the bracketed form counts, so prose cannot be mistaken for a source."""
    assert cited_pages("as described on page 37 of the manual") == set()
