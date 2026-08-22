"""
Builds the VMPP-month panel and computes the two headline diagnostics
(persistence, onset) that the plan's difficulty framing rests on.

Grain note: operates at VMPP (pack) level, the grain the raw archive has.
The plan's actual target label is at *chemical* level -- that aggregation
needs a VMPP -> chemical mapping, not present in this file. That's the
next real task after this one, not solved here.
"""

import pandas as pd


def build_pack_month_panel(df: pd.DataFrame) -> pd.DataFrame:
    all_months = pd.date_range(df["Month"].min(), df["Month"].max(), freq="MS")
    all_vmpp = df["VMPP SNOMED code"].unique()

    panel = pd.MultiIndex.from_product(
        [all_vmpp, all_months], names=["vmpp", "month"]
    ).to_frame(index=False)

    on_concession = set(zip(df["VMPP SNOMED code"], df["Month"]))
    panel["on_concession"] = panel.apply(
        lambda r: (r["vmpp"], r["month"]) in on_concession, axis=1
    )
    return panel.sort_values(["vmpp", "month"]).reset_index(drop=True)


def compute_persistence_and_onset(panel: pd.DataFrame) -> dict:
    p = panel.copy()
    p["next_month_on"] = p.groupby("vmpp")["on_concession"].shift(-1)
    transitions = p.dropna(subset=["next_month_on"])

    on_now = transitions[transitions["on_concession"]]
    clear_now = transitions[~transitions["on_concession"]]

    return {
        "persistence_rate": on_now["next_month_on"].mean(),
        "onset_rate_pack_level": clear_now["next_month_on"].mean(),
        "n_persistence_transitions": len(on_now),
        "n_onset_transitions": len(clear_now),
    }


if __name__ == "__main__":
    df = pd.read_excel("data/raw/concessions/archive.xlsx")
    panel = build_pack_month_panel(df)
    stats = compute_persistence_and_onset(panel)
    print(stats)
    panel.to_parquet("data/interim/pack_month_panel.parquet")
    print(f"\nSaved {len(panel):,} rows to data/interim/pack_month_panel.parquet")