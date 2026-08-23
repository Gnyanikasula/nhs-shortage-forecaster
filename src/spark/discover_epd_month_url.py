"""
Auto-discovers the raw EPD bulk CSV download URL for a given month by
scraping NHSBSA's dataset page, rather than relying on a manually
maintained dict of hardcoded resource UUIDs (download_epd_month.py's
KNOWN_MONTH_URLS -- fine for months looked up by hand, but a scheduled
unattended job needs to find NEW months on its own, since NHSBSA assigns
each month's file a new UUID that can't be derived from the date).

This is the same pattern already used in load_archive_xlsx.py for the
concession archive spreadsheet -- applied here to the EPD dataset page
so the download step can run genuinely unattended.
"""
import re
import requests
import bs4

DATASET_PAGE = "https://opendata.nhsbsa.net/dataset/english-prescribing-dataset-epd-with-snomed-code"
HEADERS = {"User-Agent": "student-portfolio-project (contact: replace-with-your-email)"}


def find_month_url(yyyymm: str) -> str | None:
    """Scrapes the dataset page for a resource link matching this month.
    Returns None (not an error) if the month isn't published yet --
    that's an expected, normal state for a scheduled job to encounter,
    not a failure."""
    resp = requests.get(DATASET_PAGE, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = bs4.BeautifulSoup(resp.text, "lxml")

    # Resource links on this page are named like "EPD_SNOMED_202509" or
    # similar -- match on the yyyymm string appearing in the link text
    # or href, rather than assuming one exact naming convention (NHSBSA
    # has changed link text formatting before).
    for link in soup.find_all("a", href=True):
        text = link.get_text(" ", strip=True)
        href = link["href"]
        if yyyymm in text or yyyymm in href:
            if href.startswith("/"):
                href = "https://opendata.nhsbsa.net" + href
            return href

    return None


def get_latest_published_month() -> str:
    """Finds the most recent month NHSBSA has actually published, by
    scanning the dataset page for all YYYYMM-shaped resource references
    and returning the largest. Used so the scheduled job knows what to
    even attempt, rather than guessing at "today's month minus 2"."""
    resp = requests.get(DATASET_PAGE, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = bs4.BeautifulSoup(resp.text, "lxml")

    found_months = set()
    pattern = re.compile(r"20(1[4-9]|2[0-9])(0[1-9]|1[0-2])")  # 201401-202912 range
    for link in soup.find_all("a", href=True):
        combined_text = link.get_text(" ", strip=True) + " " + link["href"]
        for m in pattern.finditer(combined_text):
            found_months.add(m.group())

    if not found_months:
        raise RuntimeError(
            "Could not find any YYYYMM-shaped resource references on the "
            "dataset page -- the page layout has likely changed. This needs "
            "a human to look at the page before the scheduled job can "
            "resume safely."
        )
    return max(found_months)


if __name__ == "__main__":
    latest = get_latest_published_month()
    print(f"Latest published month found: {latest}")
    url = find_month_url(latest)
    print(f"URL: {url}")