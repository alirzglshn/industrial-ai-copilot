from pathlib import Path

import pytest

from copilot.ingestion.parser import (
    PdfDocumentParser,
    PdfParserConfig,
    _serialize_table,
    clean_extracted_text,
    is_tabular,
)
from tests.pdf_fixtures import DIAGRAM_LABEL, build_manual_pdf

CONFIG = PdfParserConfig(image_dir=Path("unused"))


@pytest.fixture
def parser(tmp_path: Path) -> PdfDocumentParser:
    return PdfDocumentParser(PdfParserConfig(image_dir=tmp_path / "images"))


def test_serialize_table_renders_rows_and_skips_empty() -> None:
    table = [["Component", "Max Temp"], ["Pump A", "80 C"], [None, None], ["Pump B", "95 C"]]
    assert _serialize_table(table) == "Component | Max Temp\nPump A | 80 C\nPump B | 95 C"


def test_serialize_table_flattens_newlines_in_cells() -> None:
    assert _serialize_table([["Max\nTemp", "80 C"]]) == "Max Temp | 80 C"


def test_parses_every_page(parser: PdfDocumentParser, manual_pdf: Path) -> None:
    document = parser.parse(str(manual_pdf), "doc-1")

    assert document.document_id == "doc-1"
    assert document.filename == "manual.pdf"
    assert [p.page_number for p in document.pages] == [1, 2]


def test_extracts_page_text(parser: PdfDocumentParser, manual_pdf: Path) -> None:
    document = parser.parse(str(manual_pdf), "doc-1")

    page_one = document.pages[0].text
    assert "Thermal Management" in page_one
    assert "insufficient cooling airflow" in page_one
    # Text must stay on its own page or citations would point at the wrong one.
    assert "Maintenance Intervals" not in page_one
    assert "Maintenance Intervals" in document.pages[1].text


def test_extracts_ruled_table(parser: PdfDocumentParser, manual_pdf: Path) -> None:
    document = parser.parse(str(manual_pdf), "doc-1")

    tables = document.pages[1].tables
    assert len(tables) == 1
    assert "Component | Max Temp | Airflow" in tables[0]
    assert "Pump A | 80 C | 12 m3/h" in tables[0]
    assert "Pump B | 95 C | 18 m3/h" in tables[0]
    assert document.pages[0].tables == []


def test_table_content_is_not_duplicated_in_the_page_text(
    parser: PdfDocumentParser, manual_pdf: Path
) -> None:
    document = parser.parse(str(manual_pdf), "doc-1")
    page_two = document.pages[1]

    # The table is emitted once, as a table. Leaving its rows in the prose text
    # too would let one fact occupy two of the top-k retrieval slots.
    assert "Pump A" in page_two.tables[0]
    assert "Pump A" not in page_two.text
    assert "12 m3/h" not in page_two.text
    # Surrounding prose on the same page must survive the exclusion.
    assert "Maintenance Intervals" in page_two.text
    assert "cooling requirements" in page_two.text


def test_extracts_and_writes_images(parser: PdfDocumentParser, manual_pdf: Path) -> None:
    document = parser.parse(str(manual_pdf), "doc-1")

    images = document.pages[0].images
    assert len(images) == 1
    assert images[0].page_number == 1
    written = Path(images[0].storage_path)
    assert written.exists() and written.stat().st_size > 0
    assert written.suffix == ".png"
    assert document.pages[1].images == []


def test_filters_out_images_below_minimum_size(tmp_path: Path) -> None:
    pdf_path = build_manual_pdf(tmp_path / "with_logo.pdf", include_tiny_image=True)

    keeps_all = PdfDocumentParser(
        PdfParserConfig(image_dir=tmp_path / "all", min_image_width=1, min_image_height=1)
    ).parse(str(pdf_path), "doc-all")
    filtered = PdfDocumentParser(PdfParserConfig(image_dir=tmp_path / "filtered")).parse(
        str(pdf_path), "doc-filtered"
    )

    assert len(keeps_all.pages[0].images) == 2
    assert len(filtered.pages[0].images) == 1


def test_images_are_written_under_the_document_id(parser: PdfDocumentParser, manual_pdf: Path) -> None:
    document = parser.parse(str(manual_pdf), "doc-42")

    storage_path = Path(document.pages[0].images[0].storage_path)
    assert storage_path.parent.name == "doc-42"


class TestCleanExtractedText:
    def test_strips_unresolvable_glyph_ids(self) -> None:
        assert clean_extracted_text("Max (cid:11)temp(cid:12) 80 C") == "Max temp 80 C"

    def test_drops_a_line_that_was_entirely_glyph_ids(self) -> None:
        cleaned = clean_extracted_text("Real text\n(cid:11)(cid:12)(cid:13)\nMore text")
        assert "cid" not in cleaned
        assert "Real text" in cleaned and "More text" in cleaned

    def test_preserves_paragraph_breaks_the_chunker_relies_on(self) -> None:
        assert clean_extracted_text("First para.\n\nSecond para.") == "First para.\n\nSecond para."

    def test_collapses_runs_of_blank_lines(self) -> None:
        assert clean_extracted_text("A\n\n\n\nB") == "A\n\nB"


class TestIsTabular:
    """The gate that keeps diagram frames and chart axes out of the table index.

    Cases mirror grids actually observed in real Grundfos manuals.
    """

    def test_rejects_an_empty_diagram_frame(self) -> None:
        assert not is_tabular([["", ""], ["", ""]], CONFIG)

    def test_rejects_a_frame_holding_a_single_label(self) -> None:
        assert not is_tabular([["ELCB (GFCI)", "", ""], ["", "", ""]], CONFIG)

    def test_rejects_a_sparse_chart_axis(self) -> None:
        # Observed on page 7 of the CR manual: 5 filled cells out of 64.
        rows = [["" for _ in range(8)] for _ in range(8)]
        rows[0][0], rows[1][0], rows[1][5] = "[%]", "00 90 80", "2"
        assert not is_tabular(rows, CONFIG)

    def test_rejects_a_two_row_numeric_axis(self) -> None:
        # Observed on page 19 of the UPS3 manual: well filled, but no labels.
        rows = [
            ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"],
            ["0.1", "0.2", "0.3", "", "0.5", "0.6", "0.7", ""],
        ]
        assert not is_tabular(rows, CONFIG)

    def test_rejects_figure_callout_numbers(self) -> None:
        assert not is_tabular([["1 2 3 4 5 6", "", ""], ["7 8 9 10", "", ""]], CONFIG)

    def test_accepts_a_labelled_specification_table(self) -> None:
        rows = [
            ["Component", "Max Temp", "Airflow"],
            ["Pump A", "80 C", "12 m3/h"],
            ["Pump B", "95 C", "18 m3/h"],
        ]
        assert is_tabular(rows, CONFIG)

    def test_accepts_a_substantial_unlabelled_numeric_table(self) -> None:
        rows = [["1.0", "2.0", "3.0"], ["4.0", "5.0", "6.0"], ["7.0", "8.0", "9.0"]]
        assert is_tabular(rows, CONFIG)

    def test_rejects_a_grid_that_is_only_glyph_ids(self) -> None:
        # Observed on page 5 of the CMBE manual: structurally fine, no readable text.
        rows = [["(cid:11)", "(cid:12)", "(cid:13)"], ["(cid:14)", "(cid:15)", "(cid:16)"]]
        assert not is_tabular(rows, CONFIG)


def test_diagram_frame_is_not_a_table_but_keeps_its_text(tmp_path: Path) -> None:
    """Regression: rejecting a grid must not delete the text inside it.

    Table regions are excluded from the page text to avoid indexing a table
    twice, so that exclusion has to apply only to tables actually kept.
    """
    pdf_path = build_manual_pdf(tmp_path / "framed.pdf", include_diagram_frame=True)
    document = PdfDocumentParser(PdfParserConfig(image_dir=tmp_path / "img")).parse(
        str(pdf_path), "doc-framed"
    )
    page_one = document.pages[0]

    assert page_one.tables == []
    assert DIAGRAM_LABEL in page_one.text


def test_rejects_a_file_that_is_not_a_pdf(parser: PdfDocumentParser, tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a pdf")

    with pytest.raises(Exception):
        parser.parse(str(broken), "doc-broken")
