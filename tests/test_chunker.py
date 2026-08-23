import pytest

from copilot.ingestion.base import ExtractedPage, ParsedDocument
from copilot.ingestion.chunker import TABLE_PREFIX, TextChunker

LONG_TEXT = " ".join(f"Sentence number {i} describes a maintenance step." for i in range(80))


def test_empty_text_produces_no_chunks() -> None:
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    assert chunker.chunk_page(page_number=1, text="   ") == []


def test_short_text_is_a_single_chunk() -> None:
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_page(page_number=3, text="The pump overheats when airflow drops.")

    assert len(chunks) == 1
    assert chunks[0].page_number == 3
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "The pump overheats when airflow drops."


def test_long_text_splits_into_multiple_bounded_chunks() -> None:
    chunk_size = 200
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=40)
    chunks = chunker.chunk_page(page_number=1, text=LONG_TEXT)

    assert len(chunks) > 1
    # Chunks may exceed the target only by the overlap carried into them.
    assert all(len(c.text) <= chunk_size + 40 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap() -> None:
    chunker = TextChunker(chunk_size=200, chunk_overlap=60)
    chunks = chunker.chunk_page(page_number=1, text=LONG_TEXT)

    assert len(chunks) > 1
    first_tail_words = set(chunks[0].text.split())
    second_head_words = chunks[1].text.split()
    # The start of a chunk should repeat content from the end of the previous one.
    assert any(word in first_tail_words for word in second_head_words[:5])


def test_no_content_is_dropped_when_splitting() -> None:
    chunker = TextChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk_page(page_number=1, text=LONG_TEXT)

    joined = " ".join(c.text for c in chunks)
    for i in (0, 40, 79):
        assert f"Sentence number {i} " in joined


def test_paragraph_boundaries_are_preferred() -> None:
    text = "First paragraph about cooling.\n\nSecond paragraph about bearings."
    chunker = TextChunker(chunk_size=40, chunk_overlap=0)
    chunks = chunker.chunk_page(page_number=1, text=text)

    assert [c.text for c in chunks] == [
        "First paragraph about cooling.",
        "Second paragraph about bearings.",
    ]


def test_tables_become_their_own_prefixed_chunks() -> None:
    chunker = TextChunker(chunk_size=400, chunk_overlap=0)
    chunks = chunker.chunk_page(
        page_number=7,
        text="Cooling specifications follow.",
        tables=["Component | Max Temp\nPump A | 80 C"],
    )

    prose = [c for c in chunks if not c.text.startswith(TABLE_PREFIX)]
    tables = [c for c in chunks if c.text.startswith(TABLE_PREFIX)]

    assert len(prose) == 1
    assert len(tables) == 1
    assert "Pump A | 80 C" in tables[0].text
    # Prose and table content must not be merged into one chunk.
    assert "Cooling specifications" not in tables[0].text


def test_chunk_indices_are_document_global_and_pages_never_merge() -> None:
    document = ParsedDocument(
        document_id="doc-1",
        filename="manual.pdf",
        pages=[
            ExtractedPage(page_number=1, text=LONG_TEXT),
            ExtractedPage(page_number=2, text=LONG_TEXT),
        ],
    )
    chunker = TextChunker(chunk_size=200, chunk_overlap=40)
    chunks = chunker.chunk_document(document)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page_number for c in chunks} == {1, 2}
    # A chunk belongs to exactly one page, which is what makes citations exact.
    page_one_indices = [c.chunk_index for c in chunks if c.page_number == 1]
    page_two_indices = [c.chunk_index for c in chunks if c.page_number == 2]
    assert max(page_one_indices) < min(page_two_indices)


def test_overlap_is_clamped_below_chunk_size() -> None:
    chunker = TextChunker(chunk_size=100, chunk_overlap=500)
    assert chunker.chunk_overlap == 99
    # Must still terminate rather than repeating content forever.
    assert len(chunker.chunk_page(page_number=1, text=LONG_TEXT)) > 1


def test_invalid_chunk_size_rejected() -> None:
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0)
