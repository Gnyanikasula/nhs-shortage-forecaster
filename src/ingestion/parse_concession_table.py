"""
Parses a single month's price concession table (as scraped markdown/HTML text
from a CPE page) into a tidy DataFrame: one row per (drug, pack_size) line.

This module is deliberately pure: text in, DataFrame out, no network calls.
That's what makes it unit-testable against saved fixtures instead of live
HTTP requests, and it's why it's split out from the actual scraper/downloader.
"""

import re
import pandas as pd

MONTH_HEADER_RE = re.compile(
    r"granted for\s+([A-Za-z]+)\s+(\d{4})", re.IGNORECASE
)

ROW_RE = re.compile(
    r"^\|\s*(?P<drug>[^|]+?)\s*\|\s*(?P<pack>[\d.]+)\s*\|\s*"
    r"£(?P<price>[\d,]+\.\d{2})(?:\s*\(previously[^)]*\))?\s*\|\s*$"
)

TOTAL_RE = re.compile(r"to\s+\*?\*?(\d+)\*?\*?\s*\.?\s*$", re.IGNORECASE)


def parse_month_block(text: str, expected_month: str = None, expected_year: int = None) -> pd.DataFrame:
    month, year = expected_month, expected_year
    if month is None or year is None:
        header_match = MONTH_HEADER_RE.search(text)
        if not header_match:
            raise ValueError(
                "Could not find a 'granted for <Month> <Year>' header and "
                "no explicit month/year was passed. Refusing to guess."
            )
        month, year = header_match.group(1), int(header_match.group(2))

    rows = []
    for line in text.splitlines():
        m = ROW_RE.match(line.strip())
        if not m:
            continue
        drug = m.group("drug").strip()
        if drug.lower() in {"drug", "**drug**"}:
            continue
        rows.append(
            {
                "drug": drug,
                "pack_size": float(m.group("pack")),
                "price_gbp": float(m.group("price").replace(",", "")),
            }
        )

    if not rows:
        raise ValueError(
            f"Parsed zero rows for {month} {year}. The page structure has "
            "likely changed and this parser needs updating -- fail loudly, "
            "do not silently return an empty month."
        )

    df = pd.DataFrame(rows)
    df["month"] = pd.Timestamp(year=year, month=_month_num(month), day=1)

    total_match = TOTAL_RE.search(text)
    if total_match:
        stated_total = int(total_match.group(1))
        parsed_total = len(df)
        if parsed_total != stated_total:
            raise ValueError(
                f"{month} {year}: parsed {parsed_total} concession lines but "
                f"the source states {stated_total}. Verify before trusting "
                f"this month."
            )

    return df


def _month_num(name: str) -> int:
    months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ]
    return months.index(name.strip().lower()) + 1


if __name__ == "__main__":
    import sys
    from pathlib import Path

    fixture = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if fixture is None:
        print("Usage: python parse_concession_table.py <path_to_month_text>")
        sys.exit(1)

    # text = fixture.read_text()
    text = fixture.read_text(encoding="utf-8")
    df = parse_month_block(text)
    print(df.head())
    print(f"\nParsed {len(df)} rows for {df['month'].iloc[0].strftime('%B %Y')}")
    print(f"Unique drugs (pre chemical-normalisation): {df['drug'].nunique()}")