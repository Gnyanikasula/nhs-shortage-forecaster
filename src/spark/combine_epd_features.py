"""
Phase 3: combines the per-month chemical-level feature CSVs (produced by
run_full_pipeline.py, one file per month) into a single file for the
as-of join and walk-forward comparison.
"""
import glob
import pandas as pd

FEATURES_DIR = "data/interim/epd_prescribing_features"
OUTPUT_PATH = "data/interim/epd_prescribing_features_combined.csv"

if __name__ == "__main__":
    files = sorted(glob.glob(f"{FEATURES_DIR}/features_*.csv"))
    if not files:
        raise FileNotFoundError(f"No features_*.csv files found in {FEATURES_DIR}")

    print(f"Found {len(files)} monthly feature files:")
    for f in files:
        print(f"  {f}")

    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)

    dupes = combined.duplicated(subset=["BNF_CHEMICAL_SUBSTANCE_CODE", "YEAR_MONTH"]).sum()
    if dupes > 0:
        raise ValueError(f"{dupes} duplicate (chemical, month) rows found after combining -- "
                          f"investigate before using this file, don't silently keep duplicates.")

    # combined["month"] = pd.to_datetime(combined["YEAR_MONTH"], format="%Y%m")
    combined["month"] = pd.to_datetime(combined["YEAR_MONTH"], format="%Y-%m")

    print(f"\nCombined: {len(combined)} rows, {combined['BNF_CHEMICAL_SUBSTANCE'].nunique()} chemicals, "
          f"{combined['month'].nunique()} months")
    print(f"Date range: {combined['month'].min()} to {combined['month'].max()}")

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")