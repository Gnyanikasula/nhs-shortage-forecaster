"""
Phase 2 PILOT -- scoped to the 3 ADHD chemicals that actually have any
concession history (Atomoxetine, Dexamfetamine, Methylphenidate;
Lisdexamfetamine and Guanfacine never appear in the concession archive at
all, so there's no onset label to predict for them -- they're excluded,
not silently zero-filled).

CAVEAT: across these 3 chemicals there are only 8 onset events in the
entire 79-month history -- this pilot's result was directional only, and
was superseded by the full-scale test in phase2_full_scale_comparison.py
(910 onset events). Kept in the repo as the intermediate step that led
there, not as the final Phase 2 answer.

CHEMICAL NAME MISMATCH: EPD names chemicals with their salt form
("Atomoxetine hydrochloride"); our concession-derived chemical field
strips this ("Atomoxetine"). Mapped by hand below.

PUBLICATION LAG: NHSBSA's EPD data is published roughly 2 months after
the month it covers. A feature "as of month M" can only legitimately use
EPD data through month M-2.
"""

import pandas as pd
import numpy as np

EPD_PUBLICATION_LAG_MONTHS = 2

CONCESSION_TO_EPD_NAME = {
    "Atomoxetine": "Atomoxetine hydrochloride",
    "Dexamfetamine": "Dexamfetamine sulfate",
    "Methylphenidate": "Methylphenidate hydrochloride",
}


def load_epd_prescribing_features(epd_path: str) -> pd.DataFrame:
    epd = pd.read_csv(epd_path)
    epd["month"] = pd.to_datetime(epd["YEAR_MONTH"], format="%Y%m")
    epd = epd.sort_values(["chemical_name", "month"])
    epd["items_mom_growth"] = epd.groupby("chemical_name")["total_items"].pct_change()
    return epd[["month", "chemical_name", "total_items", "items_mom_growth"]]


def as_of_join(features: pd.DataFrame, epd: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    df["epd_chemical_name"] = df["chemical"].map(CONCESSION_TO_EPD_NAME)
    df = df[df["epd_chemical_name"].notna()].copy()

    df["epd_lookup_month"] = df["month"] - pd.DateOffset(months=EPD_PUBLICATION_LAG_MONTHS)

    merged = df.merge(
        epd,
        left_on=["epd_lookup_month", "epd_chemical_name"],
        right_on=["month", "chemical_name"],
        how="left",
        suffixes=("", "_epd"),
    )
    return merged


if __name__ == "__main__":
    features = pd.read_parquet("data/interim/chemical_features.parquet")
    epd = load_epd_prescribing_features("data/interim/epd_adhd_full.csv")

    adhd_chemicals = list(CONCESSION_TO_EPD_NAME.keys())
    adhd_features = features[features["chemical"].isin(adhd_chemicals)].copy()

    n_positives = adhd_features["label_onset_next_month"].sum()
    n_at_risk = adhd_features["label_onset_next_month"].notna().sum()
    print(f"ADHD pilot subset: {adhd_features['chemical'].nunique()} chemicals, "
          f"{n_at_risk} at-risk rows, {n_positives:.0f} onset events total")
    print("^ this is the real sample size everything below is computed on -- keep it in view\n")

    merged = as_of_join(adhd_features, epd)
    coverage = merged["total_items"].notna().mean()
    print(f"EPD feature coverage after as-of join: {coverage:.1%} of rows have a matched prescribing value")

    print(merged[["chemical", "month", "on_concession", "total_items",
                   "items_mom_growth", "label_onset_next_month"]].tail(15).to_string(index=False))

    merged.to_parquet("data/interim/adhd_pilot_with_epd_features.parquet")
    print(f"\nSaved to data/interim/adhd_pilot_with_epd_features.parquet")