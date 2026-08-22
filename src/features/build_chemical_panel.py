"""
Collapses the pack-level (VMPP) concession panel to chemical level, which
is the grain the plan's actual onset label is defined at (a chemical
counts as "on concession" if ANY of its pack sizes are).

Derivation method: strip everything from the first digit onward in the
free-text Drug field ("Atomoxetine 40mg capsules" -> "Atomoxetine"). This
is a heuristic, not an authoritative mapping (the authoritative source
would be NHSBSA's dm+d VMPP->VTM table) -- but validated against all 896
real VMPP codes in the archive, it collapses cleanly to 377 chemicals
with only 2 codes showing any name instability over time, both explained
(one capitalisation drift, one a genuine chemical rename: Senna ->
Sennosides). Good enough to proceed on; revisit with dm+d only if a
later data pull shows this heuristic breaking down further.

For the 2 unstable codes (and any future ones), the MOST RECENT name is
taken as canonical, not the first -- recent naming is more likely to
reflect current BNF convention than whatever was used years ago.
"""

import re
import pandas as pd


def derive_chemical_name(drug: str) -> str:
    """Strip from the first digit onward, trim trailing separators."""
    m = re.search(r"\d", drug)
    if not m:
        return drug.strip()
    return drug[: m.start()].strip().rstrip("-").strip()


def build_vmpp_to_chemical_map(df: pd.DataFrame) -> pd.DataFrame:
    """One row per VMPP code -> its canonical chemical name (most recent)."""
    df = df.copy()
    df["chemical"] = df["Drug"].apply(derive_chemical_name)

    latest = (
        df.sort_values("Month")
        .groupby("VMPP SNOMED code")
        .tail(1)[["VMPP SNOMED code", "chemical"]]
        .rename(columns={"chemical": "canonical_chemical"})
    )
    return latest


def build_chemical_month_panel(df: pd.DataFrame, vmpp_to_chem: pd.DataFrame) -> pd.DataFrame:
    """A chemical is 'on concession' in a month if ANY of its VMPP packs are."""
    df = df.merge(vmpp_to_chem, on="VMPP SNOMED code", how="left")
    assert df["canonical_chemical"].isna().sum() == 0, "Unmapped VMPP codes found -- investigate before proceeding"

    all_months = pd.date_range(df["Month"].min(), df["Month"].max(), freq="MS")
    all_chemicals = df["canonical_chemical"].unique()

    panel = pd.MultiIndex.from_product(
        [all_chemicals, all_months], names=["chemical", "month"]
    ).to_frame(index=False)

    on_concession = set(zip(df["canonical_chemical"], df["Month"]))
    panel["on_concession"] = panel.apply(
        lambda r: (r["chemical"], r["month"]) in on_concession, axis=1
    )
    return panel.sort_values(["chemical", "month"]).reset_index(drop=True)


def compute_persistence_and_onset(panel: pd.DataFrame) -> dict:
    p = panel.copy()
    p["next_month_on"] = p.groupby("chemical")["on_concession"].shift(-1)
    transitions = p.dropna(subset=["next_month_on"])

    on_now = transitions[transitions["on_concession"]]
    clear_now = transitions[~transitions["on_concession"]]

    return {
        "persistence_rate": on_now["next_month_on"].mean(),
        "onset_rate_chemical_level": clear_now["next_month_on"].mean(),
        "n_persistence_transitions": len(on_now),
        "n_onset_transitions": len(clear_now),
    }


if __name__ == "__main__":
    df = pd.read_excel("data/raw/concessions/archive.xlsx")

    vmpp_to_chem = build_vmpp_to_chemical_map(df)
    print(f"Mapped {df['VMPP SNOMED code'].nunique()} VMPP codes to "
          f"{vmpp_to_chem['canonical_chemical'].nunique()} chemicals")

    panel = build_chemical_month_panel(df, vmpp_to_chem)
    stats = compute_persistence_and_onset(panel)
    print(stats)

    panel.to_parquet("data/interim/chemical_month_panel.parquet")
    vmpp_to_chem.to_csv("data/interim/vmpp_to_chemical_map.csv", index=False)
    print(f"\nSaved {len(panel):,} rows to data/interim/chemical_month_panel.parquet")
    print(f"Saved mapping to data/interim/vmpp_to_chemical_map.csv")