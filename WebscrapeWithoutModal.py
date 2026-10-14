import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin
from xml.sax.saxutils import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_URL = (
    "https://agritech.tnau.ac.in/"
    "expert_system/paddy/Schemes.html"
)
URL = BASE_URL + "#sls"

OUTPUT_DIR = Path("output")
OUTPUT_FILE = OUTPUT_DIR / "TNAU_Paddy_Schemes.pdf"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
}

REQUEST_TIMEOUT = 45
TABLE_CHUNK_SIZE = 15


# --------------------------------------------------
# Download and parse webpage
# --------------------------------------------------

def download_page():
    print("Downloading webpage...")

    response = requests.get(
        BASE_URL,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()

    print(f"HTTP status: {response.status_code}")

    return BeautifulSoup(response.content, "html.parser")


def find_content(soup):
    """
    Try to locate the #sls section.
    Fall back to the complete main page if necessary.
    """

    anchor = soup.find(id="sls")

    if anchor is None:
        anchor = soup.find("a", attrs={"name": "sls"})

    if anchor is None:
        print(
            "Anchor #sls was not found. "
            "The complete page will be exported."
        )
        return soup.body or soup

    # Prefer a containing section or div.
    container = anchor.find_parent(["section", "div"])

    if container:
        print("Found the target anchor inside a container.")
        return container

    # Legacy pages frequently use HTML tables for layout.
    table = anchor.find_parent("table")

    if table:
        print("Found the target anchor inside a table.")
        return table

    return anchor.parent or soup


def clean_content(content):
    for element in content.select(
        "script, style, noscript, iframe"
    ):
        element.decompose()


# --------------------------------------------------
# PDF styles
# --------------------------------------------------

def create_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="DocumentTitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        spaceAfter=8 * mm,
    ))

    styles.add(ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        spaceAfter=4,
        splitLongWords=1,
        wordWrap="LTR",
    ))

    styles.add(ParagraphStyle(
        name="TableCellCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        spaceAfter=0,
        splitLongWords=1,
        wordWrap="LTR",
    ))

    styles.add(ParagraphStyle(
        name="TableHeaderCustom",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=9,
        textColor=colors.black,
        splitLongWords=1,
        wordWrap="LTR",
    ))

    return styles


# --------------------------------------------------
# Convert HTML table to paginated PDF tables
# --------------------------------------------------

def extract_table(table, styles):
    """
    Convert an HTML table into smaller ReportLab tables.
    Large tables are split into chunks so they can span pages.
    """

    rows = []

    for tr in table.find_all("tr"):
        # Avoid rows belonging to nested tables.
        if tr.find_parent("table") is not table:
            continue

        cells = [
            cell
            for cell in tr.find_all(["th", "td"])
            if cell.find_parent("tr") is tr
            and cell.find_parent("table") is table
        ]

        if not cells:
            continue

        row = []

        for cell in cells:
            text = cell.get_text(" ", strip=True)
            safe_text = escape(text) if text else " "

            # Treat the first row and explicit TH cells as headers.
            cell_style = (
                styles["TableHeaderCustom"]
                if cell.name == "th"
                else styles["TableCellCustom"]
            )

            row.append(Paragraph(safe_text, cell_style))

        rows.append(row)

    if not rows:
        return []

    # Normalize column counts.
    column_count = max(len(row) for row in rows)

    for row in rows:
        while len(row) < column_count:
            row.append(
                Paragraph(" ", styles["TableCellCustom"])
            )

    # Use the first row as the repeating header.
    header = rows[0]
    body_rows = rows[1:]

    chunks = []

    for start in range(0, len(body_rows), TABLE_CHUNK_SIZE):
        chunk = [header] + body_rows[
            start:start + TABLE_CHUNK_SIZE
        ]

        pdf_table = Table(
            chunk,
            repeatRows=1,
            hAlign="LEFT",
            splitByRow=1,
            splitInRow=1,
        )

        pdf_table.setStyle(TableStyle([
            (
                "BACKGROUND",
                (0, 0), (-1, 0),
                colors.HexColor("#DCE8D5")
            ),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))

        chunks.append(pdf_table)

    return chunks


# --------------------------------------------------
# HTML content to PDF flowables
# --------------------------------------------------

def convert_content(content, styles):
    story = []

    # Prevent duplicate extraction of nested tables.
    handled_tables = set()

    for element in content.descendants:
        if not isinstance(element, Tag):
            continue

        if element.name in [
            "h1", "h2", "h3", "h4", "h5", "h6"
        ]:
            # Avoid headings embedded in table layout.
            if element.find_parent("table"):
                continue

            text = element.get_text(" ", strip=True)

            if text:
                story.append(Paragraph(
                    escape(text),
                    styles["Heading2"]
                ))
                story.append(Spacer(1, 2 * mm))

        elif element.name == "p":
            if element.find_parent("table"):
                continue

            text = element.get_text(" ", strip=True)

            if text:
                story.append(Paragraph(
                    escape(text),
                    styles["BodyCustom"]
                ))

        elif element.name in ["ul", "ol"]:
            if element.find_parent(["ul", "ol", "table"]):
                continue

            for li in element.find_all("li", recursive=False):
                text = li.get_text(" ", strip=True)

                if text:
                    story.append(Paragraph(
                        "&#8226; " + escape(text),
                        styles["BodyCustom"]
                    ))

        elif element.name == "table":
            if id(element) in handled_tables:
                continue

            # Skip nested tables; their content is handled by
            # the outer table's cell text.
            if element.find_parent("table"):
                continue

            handled_tables.add(id(element))

            tables = extract_table(element, styles)

            if tables:
                story.append(Spacer(1, 3 * mm))

                for pdf_table in tables:
                    story.append(pdf_table)
                    story.append(Spacer(1, 3 * mm))

    # Fallback for pages without recognizable structural content.
    if not story:
        text = content.get_text("\n", strip=True)

        for line in text.splitlines():
            line = line.strip()

            if line:
                story.append(Paragraph(
                    escape(line),
                    styles["BodyCustom"]
                ))
                story.append(Spacer(1, 2 * mm))

    return story


# --------------------------------------------------
# PDF footer
# --------------------------------------------------

def add_page_number(canvas, doc):
    canvas.saveState()

    page_width, _ = landscape(A4)

    canvas.setFont("Helvetica", 7)

    canvas.drawString(
        12 * mm,
        8 * mm,
        "Source: agritech.tnau.ac.in"
    )

    canvas.drawRightString(
        page_width - 12 * mm,
        8 * mm,
        f"Page {doc.page}"
    )

    canvas.restoreState()


# --------------------------------------------------
# Main program
# --------------------------------------------------

def main():
    try:
        soup = download_page()

        page_title = (
            soup.title.get_text(" ", strip=True)
            if soup.title
            else "TNAU Paddy Schemes"
        )

        content = find_content(soup)
        clean_content(content)

        styles = create_styles()

        output_path = Path(OUTPUT_FILE).resolve()

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=landscape(A4),
            rightMargin=12 * mm,
            leftMargin=12 * mm,
            topMargin=12 * mm,
            bottomMargin=15 * mm,
            title=page_title,
            author="TNAU webpage scraper",
            allowSplitting=1,
        )

        story = [
            Paragraph(
                escape(page_title),
                styles["DocumentTitleCustom"]
            ),
            Paragraph(
                'Source: <link href="' + escape(URL)
                + '" color="blue">' + escape(URL) + "</link>",
                styles["BodyCustom"]
            ),
            Spacer(1, 5 * mm),
        ]

        story.extend(convert_content(content, styles))

        if not story:
            raise RuntimeError(
                "No readable content was found on the webpage."
            )

        print("Generating PDF...")

        doc.build(
            story,
            onFirstPage=add_page_number,
            onLaterPages=add_page_number,
        )

        print("\nPDF created successfully!")
        print(f"Saved to: {output_path}")

    except requests.exceptions.RequestException as exc:
        print(f"Network error: {exc}")

    except Exception as exc:
        print(f"Error creating PDF: {exc}")
        raise


if __name__ == "__main__":
    main()
