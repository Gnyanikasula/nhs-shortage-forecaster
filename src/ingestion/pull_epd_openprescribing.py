"""
Pulls monthly national prescribing volume (ITEMS) for the 5 ADHD chemicals
from OpenPrescribing's public API, which serves pre-aggregated national
totals per BNF chemical code -- no need to download NHSBSA's raw ~17M-row
monthly files just to get 5 numbers out of each one.

This REPLACES pull_epd_bulk.py and the datastore_search calls in
pull_epd_snomed.py for the Mar 2025 - Jun 2026 gap. Keep epd_adhd.csv
(the original file, Nov 2020 - Feb 2025) as-is; this only fills the tail.

Data source: openprescribing.net/api/ (Bennett Institute, University of
Oxford), built on top of the same NHSBSA EPD data, just already aggregated.
"""
import argparse
import requests
import pandas as pd

API_BASE = "https://openprescribing.net/api/1.0/spending/"
HEADERS = {"User-Agent": "student-portfolio-project (contact: replace-with-your-email)"}

ADHD_BNF_CHEMICAL_CODES = {
    "0404000L0": "Dexamfetamine sulfate",
    "0404000M0": "Methylphenidate hydrochloride",
    "0404000S0": "Atomoxetine hydrochloride",
    "0404000U0": "Lisdexamfetamine dimesylate",
    "0404000V0": "Guanfacine",
}


def fetch_chemical(bnf_code: str) -> pd.DataFrame:
    resp = requests.get(
        API_BASE,
        params={"code": bnf_code, "format": "json"},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    records = resp.json()
    if not records:
        raise ValueError(
            f"{bnf_code}: API returned zero records. Either the code is "
            f"wrong or this chemical genuinely has no recent data -- verify "
            f"manually at openprescribing.net/analyse/ before trusting this."
        )
    df = pd.DataFrame(records)
    # date field is 'date': "2026-05-01" for May 2026 etc.
    df["YEAR_MONTH"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m").astype(int)
    df["BNF_CHEMICAL_SUBSTANCE_CODE"] = bnf_code
    df["BNF_CHEMICAL_SUBSTANCE"] = ADHD_BNF_CHEMICAL_CODES[bnf_code]
    df = df.rename(columns={"items": "total_items"})
    return df[["YEAR_MONTH", "BNF_CHEMICAL_SUBSTANCE_CODE", "BNF_CHEMICAL_SUBSTANCE", "total_items"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="epd_adhd_via_openprescribing.csv")
    parser.add_argument("--start", type=int, default=202503,
                         help="Only keep months >= this YYYYMM (API returns 5 years, we only want the gap)")
    args = parser.parse_args()

    all_chemicals = []
    for code, name in ADHD_BNF_CHEMICAL_CODES.items():
        print(f"Fetching {name} ({code}) ...")
        try:
            df = fetch_chemical(code)
            all_chemicals.append(df)
            print(f"  Got {len(df)} months total (last 5 years)")
        except Exception as e:
            print(f"  FAILED -- {e}")

    if not all_chemicals:
        print("No chemicals fetched successfully -- stopping.")
        return

    result = pd.concat(all_chemicals, ignore_index=True)
    result = result[result["YEAR_MONTH"] >= args.start].sort_values(
        ["YEAR_MONTH", "BNF_CHEMICAL_SUBSTANCE_CODE"]
    )
    result.to_csv(args.out, index=False)
    print(f"\nSaved {len(result)} rows to {args.out}")
    print(f"Months covered: {sorted(result['YEAR_MONTH'].unique())}")

    counts = result.groupby("YEAR_MONTH").size()
    short = counts[counts != 5]
    if len(short):
        print(f"\nWARNING -- months with != 5 chemical rows (incomplete):")
        print(short)


if __name__ == "__main__":
    main()