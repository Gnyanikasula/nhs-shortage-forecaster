"""
Merges the two EPD files, which unfortunately use different column names
for the same fields (inherited from their two different sources -- the
NHSBSA datastore API vs OpenPrescribing's API):

    epd_adhd.csv (Nov 2020 - Feb 2025, via NHSBSA datastore_search):
        BNF_CHEMICAL_SUBSTANCE       <- this is actually the CODE
        CHEMICAL_SUBSTANCE_BNF_DESCR <- this is the chemical NAME

    epd_adhd_recent.csv (Mar 2025 - May 2026, via OpenPrescribing API):
        BNF_CHEMICAL_SUBSTANCE_CODE  <- the CODE
        BNF_CHEMICAL_SUBSTANCE       <- the chemical NAME

Renamed to a single consistent schema before concatenating, not just
pd.concat'd blindly (which would silently produce 4 mismatched columns
full of NaNs instead of erroring -- the dangerous kind of bug).
"""
import pandas as pd

STANDARD_COLUMNS = ["YEAR_MONTH", "chemical_code", "chemical_name", "total_items"]


def load_original(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "BNF_CHEMICAL_SUBSTANCE": "chemical_code",
        "CHEMICAL_SUBSTANCE_BNF_DESCR": "chemical_name",
    })
    return df[STANDARD_COLUMNS]


def load_recent(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "BNF_CHEMICAL_SUBSTANCE_CODE": "chemical_code",
        "BNF_CHEMICAL_SUBSTANCE": "chemical_name",
    })
    return df[STANDARD_COLUMNS]


def merge(original_path: str, recent_path: str, output_path: str) -> pd.DataFrame:
    original = load_original(original_path)
    recent = load_recent(recent_path)

    overlap = set(original["YEAR_MONTH"]) & set(recent["YEAR_MONTH"])
    if overlap:
        raise ValueError(
            f"Original and recent files overlap on months {sorted(overlap)} "
            f"-- expected them to be disjoint (Nov2020-Feb2025 vs Mar2025-May2026). "
            f"Investigate before merging, don't just drop_duplicates and hope."
        )

    combined = pd.concat([original, recent], ignore_index=True)
    combined = combined.sort_values(["YEAR_MONTH", "chemical_code"]).reset_index(drop=True)

    # Self-check: every month should have exactly 5 chemical rows
    counts = combined.groupby("YEAR_MONTH").size()
    bad = counts[counts != 5]
    if len(bad):
        raise ValueError(
            f"After merging, {len(bad)} months don't have exactly 5 chemical "
            f"rows:\n{bad}\nDo not save a panel with silently incomplete months."
        )

    combined.to_csv(output_path, index=False)
    print(f"Merged {len(original)} + {len(recent)} = {len(combined)} rows")
    print(f"Full range: {combined['YEAR_MONTH'].min()} to {combined['YEAR_MONTH'].max()}")
    print(f"Total months: {combined['YEAR_MONTH'].nunique()}")
    print(f"Saved to {output_path}")
    return combined


if __name__ == "__main__":
    merge("epd_adhd.csv", "epd_adhd_recent.csv", "epd_adhd_full.csv")