
import asyncio
import csv
import re
from pathlib import Path

import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    LongTable,
    TableStyle,
    Paragraph,
    Spacer,
)

URL = "https://www.tnagrisnet.tn.gov.in/people_app/GoScheme"

OUTPUT_DIR = Path("output")
CSV_FILE = OUTPUT_DIR / "tn_agriculture_go_schemes.csv"
#PDF_FILE = OUTPUT_DIR / "tn_agriculture_go_schemes.pdf"

OUTPUT_DIR.mkdir(exist_ok=True)


async def get_table(page):
    """Find the table containing the scheme records."""
    tables = page.locator("table")
    for i in range(await tables.count()):
        table = tables.nth(i)
        if await table.locator("tbody tr").count() > 0:
            return table

    raise RuntimeError(
        "No data table found. Inspect the page's table selectors."
    )


async def set_page_size(page, table):
    """Try common table page-size dropdown selectors."""
    selects = page.locator("select")

    for i in range(await selects.count()):
        select = selects.nth(i)
        options = await select.locator("option").all_text_contents()
        values = await select.locator("option").evaluate_all(
            "(els) => els.map(e => e.value)"
        )

        # Match a dropdown offering 25 rows.
        match = None
        for j, (label, value) in enumerate(zip(options, values)):
            if label.strip() == "25" or value.strip() == "25":
                match = value
                break

        if match is not None:
            await select.select_option(match)
            await page.wait_for_timeout(1000)
            return

    print("Warning: No dropdown offering 25 was found.")


async def read_headers(table):
    headers = await table.locator("thead tr").last.locator(
        "th, td"
    ).all_text_contents()

    return [re.sub(r"\s+", " ", h).strip() for h in headers]


async def extract_modal_text(page):
    """Read visible modal/dialog text after clicking a scheme."""
    # Bootstrap, accessible dialogs and common modal libraries.
    selectors = [
        ".modal.show",
        ".modal[style*='display: block']",
        "[role='dialog']",
        ".ui-dialog",
        ".swal2-popup",
    ]

    for selector in selectors:
        dialogs = page.locator(selector)
        for i in range(await dialogs.count()):
            dialog = dialogs.nth(i)
            if await dialog.is_visible():
                text = await dialog.inner_text()
                if text.strip():
                    return re.sub(r"\n{3,}", "\n\n", text).strip()

    # Fallback: inspect visible dialog-like containers.
    return ""


async def close_modal(page):
    await page.keyboard.press("Escape")
    await page.wait_for_timeout(500)

    # Fallback if Escape does not close the dialog.
    for selector in [
        ".modal.show button.close",
        ".modal.show button.btn-close",
        "[role='dialog'] button[aria-label='Close']",
        ".ui-dialog-titlebar-close",
    ]:
        button = page.locator(selector).first
        try:
            if await button.is_visible():
                await button.click()
                await page.wait_for_timeout(500)
                break
        except Exception:
            pass


async def scrape():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        page.set_default_timeout(10000)

        try:
            await page.goto(URL, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)

            print("Page title:", await page.title())

            table = await get_table(page)
            await set_page_size(page, table)
            table = await get_table(page)

            headers = await read_headers(table)
            print("Table headers:", headers)

            # Identify the scheme column by its header.
            scheme_index = next(
                (
                    i for i, h in enumerate(headers)
                    if "scheme" in h.lower()
                ),
                None,
            )

            if scheme_index is None:
                raise RuntimeError(
                    "Could not identify the Scheme column. "
                    "Check the printed table headers."
                )

            records = []
            seen_rows = set()

            while True:
                table = await get_table(page)
                rows = table.locator("tbody tr")
                row_count = await rows.count()

                for row_index in range(row_count):
                    table = await get_table(page)
                    rows = table.locator("tbody tr")
                    row = rows.nth(row_index)

                    cells = row.locator("td")
                    cell_count = await cells.count()

                    if cell_count == 0:
                        continue

                    values = [
                        re.sub(
                            r"\s+", " ",
                            (await cells.nth(i).inner_text()).strip()
                        )
                        for i in range(cell_count)
                    ]

                    row_key = tuple(values)
                    if row_key in seen_rows:
                        continue
                    seen_rows.add(row_key)

                    # Locate the scheme link inside the scheme cell.
                    scheme_cell = cells.nth(
                        min(scheme_index, cell_count - 1)
                    )
                    links = scheme_cell.locator("a")
                    modal_text = ""

                    if await links.count() > 0:
                        try:
                            await links.first.click()
                            await page.wait_for_timeout(500)
                            modal_text = await extract_modal_text(page)

                            if not modal_text:
                                print(
                                    "Warning: Modal text not detected for:",
                                    values[scheme_index]
                                    if scheme_index < len(values)
                                    else values,
                                )

                        except PlaywrightTimeoutError:
                            print("Timed out opening scheme:", values)

                        finally:
                            await close_modal(page)

                    # Place modal details after GO & Guidelines.
                    go_index = next(
                        (
                            i for i, h in enumerate(headers)
                            if "go" in h.lower()
                            and (
                                "guideline" in h.lower()
                                or "guidelines" in h.lower()
                            )
                        ),
                        None,
                    )

                    record = {}
                    for i, value in enumerate(values):
                        header = (
                            headers[i]
                            if i < len(headers)
                            else f"Column_{i + 1}"
                        )
                        record[header or f"Column_{i + 1}"] = value

                    if go_index is not None:
                        go_header = headers[go_index]
                        ordered = {}
                        for i, (key, value) in enumerate(record.items()):
                            ordered[key] = value
                            if i == go_index:
                                ordered["Scheme Modal Details"] = modal_text
                        record = ordered
                    else:
                        record["Scheme Modal Details"] = modal_text

                    records.append(record)
                    print(f"Scraped {len(records)} rows")

                # Move to the next table page, if available.
                next_button = page.locator(
                    "a[aria-label='Next'], "
                    "button[aria-label='Next'], "
                    "li.next:not(.disabled) a, "
                    ".paginate_button.next:not(.disabled), "
                    "a:has-text('Next')"
                ).first

                try:
                    if not await next_button.is_visible():
                        break

                    classes = await next_button.get_attribute("class") or ""
                    disabled = await next_button.get_attribute("disabled")
                    aria_disabled = (
                        await next_button.get_attribute("aria-disabled")
                    )

                    if (
                        "disabled" in classes.lower()
                        or disabled is not None
                        or aria_disabled == "true"
                    ):
                        break

                    old_first_row = (
                        await get_table(page)
                    ).locator("tbody tr").first

                    old_text = await old_first_row.inner_text()
                    await next_button.click()
                    await page.wait_for_timeout(700)

                    new_text = await (
                        await get_table(page)
                    ).locator("tbody tr").first.inner_text()

                    if new_text == old_text:
                        break

                except Exception:
                    # Pagination may be absent or implemented differently.
                    break

            if not records:
                raise RuntimeError(
                    "No records were collected. Inspect the page and "
                    "adjust the table/modal selectors."
                )

            # Write UTF-8 with BOM for Excel compatibility.
            df = pd.DataFrame(records)
            df.to_csv(
                CSV_FILE,
                index=False,
                encoding="utf-8-sig",
                quoting=csv.QUOTE_ALL,
            )

            print(f"\nCSV saved: {CSV_FILE.resolve()}")
            print(f"Total records: {len(df)}")

        finally:
            await browser.close()


def convert_csv_to_pdf():
    """Convert the CSV into a landscape PDF table."""
    df = pd.read_csv(CSV_FILE, dtype=str, keep_default_na=False)

    page_width, _ = landscape(A4)
    doc = SimpleDocTemplate(
        str(PDF_FILE),
        pagesize=landscape(A4),
        rightMargin=8 * mm,
        leftMargin=8 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )

    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=6,
        leading=7.5,
        alignment=TA_LEFT,
        spaceAfter=0,
        spaceBefore=0,
        wordWrap="CJK",
    )
    header_style = ParagraphStyle(
        "TableHeader",
        parent=cell_style,
        fontName="Helvetica-Bold",
        fontSize=6.5,
        leading=8,
        textColor=colors.white,
    )

    def paragraph(value, style):
        text = str(value or "")
        text = (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br/>")
        )
        return Paragraph(text, style)

    data = [
        [paragraph(col, header_style) for col in df.columns]
    ]

    for _, row in df.iterrows():
        data.append([
            paragraph(value, cell_style) for value in row.tolist()
        ])

    column_count = max(len(df.columns), 1)
    available_width = page_width - 16 * mm
    col_widths = [available_width / column_count] * column_count

    table = LongTable(
        data,
        colWidths=col_widths,
        repeatRows=1,
        splitByRow=1,
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white, colors.HexColor("#F1F5F9")
        ]),
    ]))

    doc.build([
        Paragraph("Tamil Nadu Agriculture GO Schemes", styles["Title"]),
        Spacer(1, 5 * mm),
        Paragraph(
            f"Records exported: {len(df)}",
            styles["Normal"],
        ),
        Spacer(1, 4 * mm),
        table,
    ])

    print(f"PDF saved: {PDF_FILE.resolve()}")


if __name__ == "__main__":
    asyncio.run(scrape())
    #convert_csv_to_pdf()