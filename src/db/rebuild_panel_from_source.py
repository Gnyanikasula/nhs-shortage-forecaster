"""
Rebuilds the ENTIRE Phase 1 panel and production model from source, in
one run, with no pre-existing local state required. This is what
finally makes the pipeline genuinely ready for an ephemeral GitHub
Actions runner: everything up to this point (Docker, Postgres, the
database-backed orchestrator) still assumed chemical_features.parquet
and phase1_production_model.joblib already existed on disk. This script
is what actually PRODUCES them, from nothing, each run.

Steps, all real logic from earlier phases, combined:
  1. Download the current concession archive spreadsheet (small, ~450KB --
     nothing like the 7.6GB EPD files) by scraping CPE's archive page for
     the current link, since the filename changes monthly and can't be
     hardcoded.
  2. Build the pack-level (VMPP) month panel.
  3. Collapse to chemical level (strip-to-first-digit + most-recent-name
     canonicalisation, validated in Phase 1 to be sufficient at this
     grain -- see build_chemical_panel.py's original docstring).
  4. Build history features, including every bug fix found in Phase 1:
     the object-dtype bitwise-NOT trap (use == True, not ~), the
     leakage-safe expanding onset rate (shift(1) BEFORE cumsum), and the
     NaN fill on n_prior_valid_transitions.
  5. Train the production XGBoost model on the full resulting history.

Cost of doing this every run: seconds, not minutes -- the concession
archive is ~9,000 rows, nothing like the EPD data. This matches the
project's standing philosophy (see data/README.md): regenerate from
source rather than persist what's cheap to rebuild.
"""
import re
import numpy as np
import pandas as pd
import requests
import bs4
from xgboost import XGBClassifier

ARCHIVE_PAGE_URL = "https://cpe.org.uk/funding-and-reimbursement/reimbursement/price-concessions/archive/"
HEADERS = {"User-Agent": "student-portfolio-project (contact: replace-with-your-email)"}

FEATURE_COLUMNS = [
    "streak_length", "time_since_last_concession_filled", "n_previous_episodes",
    "n_prior_valid_transitions", "month_sin", "month_cos",
    "chemical_historical_onset_rate",
]


# Step 1: download the current archive spreadsheet 

def find_current_xlsx_url(session: requests.Session = None) -> str:
    session = session or requests.Session()
    resp = session.get(ARCHIVE_PAGE_URL, timeout=30, headers=HEADERS)
    resp.raise_for_status()
    soup = bs4.BeautifulSoup(resp.text, "lxml")
    link = soup.find("a", string=lambda s: s and "archive spreadsheet" in s.lower())
    if link is None or not link.get("href"):
        raise RuntimeError(
            "Could not find the archive spreadsheet link on the CPE archive "
            "page -- the page layout has likely changed. This needs a human "
            "to look at the page before this can run unattended again."
        )
    return link["href"]


def download_archive(dest_path: str = "/tmp/archive.xlsx") -> str:
    url = find_current_xlsx_url()
    print(f"Downloading current archive from {url}")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    print(f"  Saved {len(resp.content) / 1e6:.1f} MB to {dest_path}")
    return dest_path


# Step 2: pack-level panel

def build_pack_month_panel(df: pd.DataFrame) -> pd.DataFrame:
    all_months = pd.date_range(df["Month"].min(), df["Month"].max(), freq="MS")
    all_vmpp = df["VMPP SNOMED code"].unique()
    panel = pd.MultiIndex.from_product([all_vmpp, all_months], names=["vmpp", "month"]).to_frame(index=False)
    on_concession = set(zip(df["VMPP SNOMED code"], df["Month"]))
    panel["on_concession"] = panel.apply(lambda r: (r["vmpp"], r["month"]) in on_concession, axis=1)
    return panel.sort_values(["vmpp", "month"]).reset_index(drop=True)


# Step 3: chemical-level panel

def derive_chemical_name(drug: str) -> str:
    m = re.search(r"\d", drug)
    if not m:
        return drug.strip()
    return drug[: m.start()].strip().rstrip("-").strip()


def build_chemical_panel(archive_df: pd.DataFrame) -> pd.DataFrame:
    df = archive_df.copy()
    df["chemical"] = df["Drug"].apply(derive_chemical_name)
    vmpp_to_chem = (
        df.sort_values("Month").groupby("VMPP SNOMED code").tail(1)
        [["VMPP SNOMED code", "chemical"]].rename(columns={"chemical": "canonical_chemical"})
    )
    merged = df.merge(vmpp_to_chem, on="VMPP SNOMED code", how="left")

    all_months = pd.date_range(merged["Month"].min(), merged["Month"].max(), freq="MS")
    all_chemicals = merged["canonical_chemical"].unique()
    panel = pd.MultiIndex.from_product([all_chemicals, all_months], names=["chemical", "month"]).to_frame(index=False)
    on_concession = set(zip(merged["canonical_chemical"], merged["Month"]))
    panel["on_concession"] = panel.apply(lambda r: (r["chemical"], r["month"]) in on_concession, axis=1)
    return panel.sort_values(["chemical", "month"]).reset_index(drop=True)


# Step 4: history features (with all Phase 1 bug fixes)

def build_history_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["chemical", "month"]).reset_index(drop=True)
    prev_state = df.groupby("chemical")["on_concession"].shift(1)

    state_changed = (df["on_concession"] != prev_state) | prev_state.isna()
    streak_id = state_changed.groupby(df["chemical"]).cumsum()
    df["streak_length"] = df.groupby(["chemical", streak_id]).cumcount() + 1

    df["_month_idx"] = df["month"].dt.year * 12 + df["month"].dt.month
    df["_on_month_idx"] = np.where(df["on_concession"], df["_month_idx"], np.nan)
    df["_last_on_month_idx"] = df.groupby("chemical")["_on_month_idx"].ffill()
    df["time_since_last_concession"] = df["_month_idx"] - df["_last_on_month_idx"]

    prev_state_bool = (prev_state == True)
    episode_start = df["on_concession"] & (~prev_state_bool)
    df["n_previous_episodes"] = episode_start.groupby(df["chemical"]).cumsum()

    next_state = df.groupby("chemical")["on_concession"].shift(-1)
    is_valid_transition = (~df["on_concession"]) & next_state.notna()
    onset_event = np.where(is_valid_transition, next_state.astype(float), np.nan)
    df["_clear_transition_ind"] = is_valid_transition.astype(float)
    df["_onset_ind"] = pd.Series(onset_event, index=df.index).fillna(0) * df["_clear_transition_ind"]

    clear_shifted = df.groupby("chemical")["_clear_transition_ind"].shift(1)
    onset_shifted = df.groupby("chemical")["_onset_ind"].shift(1)
    cum_clear = clear_shifted.groupby(df["chemical"]).cumsum()
    cum_onset = onset_shifted.groupby(df["chemical"]).cumsum()

    df["chemical_historical_onset_rate"] = cum_onset / cum_clear
    df["n_prior_valid_transitions"] = cum_clear.fillna(0)
    df["month_of_year"] = df["month"].dt.month
    df["label_onset_next_month"] = np.where(is_valid_transition, next_state.astype(float), np.nan)

    df = df.drop(columns=["_month_idx", "_on_month_idx", "_last_on_month_idx",
                           "_clear_transition_ind", "_onset_ind"])
    return df


# Step 5: train the production model

def prepare_model_frame(features: pd.DataFrame):
    df = features[features["label_onset_next_month"].notna()].copy()
    df["time_since_last_concession_filled"] = df["time_since_last_concession"].fillna(999)
    df["month_sin"] = np.sin(2 * np.pi * df["month_of_year"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_of_year"] / 12)
    overall_rate = df["label_onset_next_month"].mean()
    df["chemical_historical_onset_rate"] = df["chemical_historical_onset_rate"].fillna(overall_rate)
    return df, overall_rate


def rebuild_everything() -> dict:
    """Entry point: returns {"features": DataFrame, "model_bundle": dict}
    ready for immediate scoring -- nothing written to disk that needs to
    survive between runs."""
    archive_path = download_archive()
    archive_df = pd.read_excel(archive_path)
    print(f"Archive loaded: {len(archive_df)} rows, "
          f"{archive_df['Month'].min()} to {archive_df['Month'].max()}, "
          f"{archive_df['VMPP SNOMED code'].nunique()} VMPP codes")

    chemical_panel = build_chemical_panel(archive_df)
    features = build_history_features(chemical_panel)
    print(f"Panel rebuilt: {len(features)} rows, {features['chemical'].nunique()} chemicals")

    model_df, overall_rate = prepare_model_frame(features)
    X = model_df[FEATURE_COLUMNS]
    y = model_df["label_onset_next_month"].astype(int)

    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                           eval_metric="aucpr", random_state=42)
    model.fit(X, y)
    print(f"Model trained: {len(model_df)} rows, {y.sum()} onset events")

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "fallback_historical_rate": overall_rate,
    }
    return {"features": features, "model_bundle": bundle}


if __name__ == "__main__":
    result = rebuild_everything()
    print(f"\nDone. Features shape: {result['features'].shape}")
    print(f"Model ready to score. Trained through: {result['features']['month'].max()}")