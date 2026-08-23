"""
Phase 3, Task 1: download ONE month of raw EPD (~7.6GB, ~17M rows) to the
D drive, streamed to disk (never held whole in RAM). This proves the
download path works before we scale to more months.

URL taken from the verified working pattern in pull_bulk_chemical_spending.py
(the same download that succeeded for the abandoned bulk-pull attempt --
confirmed 7.6GB / real data, not guessed). Re-derive from NHSBSA's resource
page if this exact resource ID ever 404s -- the UUID is month-specific and
not derivable from the date alone.

Usage:
    python download_epd_month.py --month 202503
"""
import argparse
import os
import requests

HEADERS = {"User-Agent": "student-portfolio-project (contact: replace-with-your-email)"}

# KNOWN_MONTH_URLS = {
#     "202503": "https://opendata.nhsbsa.net/dataset/906115a6-4155-44be-8b81-f8e83cebfb84/resource/ea287041-1027-4062-9db9-040f48223b13/download/epd_snomed_202503.csv",
#     "202504": "https://opendata.nhsbsa.net/dataset/906115a6-4155-44be-8b81-f8e83cebfb84/resource/a39bb2a2-189c-43ef-8783-2e77ccd794a0/download/epd_snomed_202504.csv",
#     "202505": "https://opendata.nhsbsa.net/dataset/906115a6-4155-44be-8b81-f8e83cebfb84/resource/ede6bcf2-5d71-437f-a3fb-fc9817d7455c/download/epd_snomed_202505.csv",
#     "202506": "https://opendata.nhsbsa.net/dataset/906115a6-4155-44be-8b81-f8e83cebfb84/resource/943c3b10-3999-475a-b6b7-d77e1fcf8e8a/download/epd_snomed_202506.csv",
# }

KNOWN_MONTH_URLS = {
    "202503": "https://opendata.nhsbsa.net/dataset/906115a6-4155-44be-8b81-f8e83cebfb84/resource/ea287041-1027-4062-9db9-040f48223b13/download/epd_snomed_202503.csv",
    "202504": "https://opendata.nhsbsa.net/dataset/906115a6-4155-44be-8b81-f8e83cebfb84/resource/a39bb2a2-189c-43ef-8783-2e77ccd794a0/download/epd_snomed_202504.csv",
    "202505": "https://opendata.nhsbsa.net/dataset/906115a6-4155-44be-8b81-f8e83cebfb84/resource/ede6bcf2-5d71-437f-a3fb-fc9817d7455c/download/epd_snomed_202505.csv",
    "202506": "https://opendata.nhsbsa.net/dataset/906115a6-4155-44be-8b81-f8e83cebfb84/resource/943c3b10-3999-475a-b6b7-d77e1fcf8e8a/download/epd_snomed_202506.csv",
    "202507": "https://opendata.nhsbsa.net/dataset/906115a6-4155-44be-8b81-f8e83cebfb84/resource/a63c5618-2af4-479c-9ae1-a3227d620ceb/download/epd_snomed_202507.csv",
    "202508": "https://opendata.nhsbsa.net/dataset/906115a6-4155-44be-8b81-f8e83cebfb84/resource/1200d488-d175-4fbd-827d-149ba65ea104/download/epd_snomed_202508.csv",
}

DEST_DIR = "data/raw/epd_bulk"


def download_month(yyyymm: str) -> str:
    if yyyymm not in KNOWN_MONTH_URLS:
        raise ValueError(
            f"No verified URL for {yyyymm}. Add it to KNOWN_MONTH_URLS -- look up "
            f"the resource ID from https://opendata.nhsbsa.net/dataset/"
            f"english-prescribing-dataset-epd-with-snomed-code, don't guess the URL."
        )
    url = KNOWN_MONTH_URLS[yyyymm]
    os.makedirs(DEST_DIR, exist_ok=True)
    dest_path = os.path.join(DEST_DIR, f"epd_snomed_{yyyymm}.csv")

    if os.path.exists(dest_path):
        size_gb = os.path.getsize(dest_path) / 1e9
        print(f"{dest_path} already exists ({size_gb:.2f} GB) -- delete it first if you want to re-download.")
        return dest_path

    print(f"Downloading {yyyymm} from {url}")
    print("This is ~7.6GB -- streaming to disk, expect several minutes depending on connection.")

    with requests.get(url, headers=HEADERS, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        total_bytes = 0
        last_reported_gb = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024 * 8):
                f.write(chunk)
                total_bytes += len(chunk)
                gb_so_far = total_bytes / 1e9
                if gb_so_far - last_reported_gb >= 0.5:
                    print(f"  ... {gb_so_far:.1f} GB downloaded")
                    last_reported_gb = gb_so_far

    final_size_gb = os.path.getsize(dest_path) / 1e9
    print(f"Done. {dest_path} -- {final_size_gb:.2f} GB")
    return dest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True, help="YYYYMM, e.g. 202503")
    args = parser.parse_args()
    download_month(args.month)