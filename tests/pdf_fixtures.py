"""Builds synthetic technical-manual PDFs so ingestion is tested against real
PDF structure (drawn text, ruled tables, embedded rasters) rather than mocks.
"""

from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

PAGE_WIDTH, PAGE_HEIGHT = LETTER

PAGE1_PARAGRAPHS = [
    "Section 4.2 Thermal Management of the Model X Circulation Pump.",
    "The pump housing is rated for continuous operation at ambient temperatures",
    "below 40 degrees Celsius. Overheating is most commonly caused by",
    "insufficient cooling airflow across the motor fins. A blocked intake filter",
    "reduces airflow by up to sixty percent and will trigger the thermal cutout.",
    "Inspect the intake filter every 500 operating hours and replace it whenever",
    "the differential pressure exceeds 0.8 bar. A secondary cause of overheating",
    "is degraded bearing lubrication, which raises friction and shaft temperature.",
    "Bearings must be regreased at the interval given in the maintenance table.",
    "If the thermal cutout trips repeatedly, verify the supply voltage is within",
    "tolerance before replacing the motor assembly. Undervoltage increases the",
    "current draw and causes the windings to heat beyond their rated limit.",
]

PAGE2_PARAGRAPHS = [
    "Section 4.3 Maintenance Intervals and Cooling Specifications.",
    "The table below lists the cooling requirements for each pump variant.",
    "Values assume a clean filter and nominal supply voltage at sea level.",
]

TABLE_ROWS = [
    ["Component", "Max Temp", "Airflow"],
    ["Pump A", "80 C", "12 m3/h"],
    ["Pump B", "95 C", "18 m3/h"],
]

DIAGRAM_LABEL = "Impeller clearance"


def _draw_paragraphs(pdf: canvas.Canvas, lines: list[str], start_y: float) -> float:
    pdf.setFont("Helvetica", 11)
    y = start_y
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 16
    return y


def _draw_table(pdf: canvas.Canvas, rows: list[list[str]], top_y: float) -> float:
    """Draws a ruled grid so pdfplumber's line-based table detection finds it."""
    col_width = 140.0
    row_height = 24.0
    left = 72.0
    n_rows = len(rows)
    n_cols = len(rows[0])
    bottom_y = top_y - n_rows * row_height

    pdf.setLineWidth(1)
    for i in range(n_rows + 1):
        y = top_y - i * row_height
        pdf.line(left, y, left + n_cols * col_width, y)
    for j in range(n_cols + 1):
        x = left + j * col_width
        pdf.line(x, top_y, x, bottom_y)

    pdf.setFont("Helvetica", 10)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            pdf.drawString(left + j * col_width + 6, top_y - (i + 1) * row_height + 8, cell)

    return bottom_y


def _image_reader(width: int, height: int) -> ImageReader:
    image = PILImage.new("RGB", (width, height), color=(30, 90, 160))
    for x in range(width):
        for y in range(min(height, 20)):
            image.putpixel((x, y), (220, 60, 40))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return ImageReader(buffer)


def _draw_diagram_frame(pdf: canvas.Canvas, label: str, x: float, y: float) -> None:
    """A ruled box holding one label.

    Line-based table detection reports these as tables, and illustrated manuals
    are full of them. The parser must reject the frame as a table while keeping
    its label in the page text.
    """
    width, height = 200.0, 60.0
    pdf.setLineWidth(1)
    pdf.rect(x, y, width, height)
    pdf.line(x, y + height / 2, x + width, y + height / 2)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(x + 6, y + height / 2 + 8, label)


def build_manual_pdf(
    path: Path,
    include_tiny_image: bool = False,
    include_diagram_frame: bool = False,
) -> Path:
    """Two-page manual: prose + a 200x200 diagram on page 1, prose + a ruled table on page 2."""
    pdf = canvas.Canvas(str(path), pagesize=LETTER)

    y = _draw_paragraphs(pdf, PAGE1_PARAGRAPHS, PAGE_HEIGHT - 72)
    pdf.drawImage(_image_reader(200, 200), 72, y - 220, width=200, height=200)
    if include_tiny_image:
        # A 16x16 raster stands in for the logos and spacers manuals repeat on
        # every page; the parser is expected to filter it out.
        pdf.drawImage(_image_reader(16, 16), 400, y - 60, width=16, height=16)
    if include_diagram_frame:
        _draw_diagram_frame(pdf, DIAGRAM_LABEL, 320, y - 200)
    pdf.showPage()

    y = _draw_paragraphs(pdf, PAGE2_PARAGRAPHS, PAGE_HEIGHT - 72)
    _draw_table(pdf, TABLE_ROWS, y - 20)
    pdf.showPage()

    pdf.save()
    return path
