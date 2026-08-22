"""
Phase 2, full scale. Joins prescribing volume (chemical_prescribing_volume.csv,
360 chemicals) onto the concession panel via the same as-of join and
publication-lag logic validated in the ADHD pilot, then runs the SAME
walk-forward comparison as Phase 1 -- history-only vs history+prescribing --
restricted to the identical set of rows for both, so the comparison is
fair. Comparing against Phase 1's original numbers (computed on a
different, larger set of rows including chemicals/months with no EPD
coverage) would not isolate the true marginal value of the EPD features.

COVERAGE: prescribing data only goes back to June 2021 (OpenPrescribing's
~5-year rolling window), and with a 2-month publication lag applied, that
means "as of month M" features are only available from M = August 2021
onward. Rows before that are excluded from BOTH sides of the comparison,
not just the EPD side -- this keeps the comparison fair, at the cost of
losing the first ~19 months of the panel.

CHEMICAL COVERAGE: 360 of 377 chemicals have prescribing data. Rows for
the other 17 (no BNF match found) are dropped from both sides too, for
the same fairness reason.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

sys.path.insert(0, "src/models")
from baselines_walk_forward import precision_at_k, BURN_IN_MONTHS

EPD_PUBLICATION_LAG_MONTHS = 2

HISTORY_FEATURE_COLUMNS = [
    "streak_length", "time_since_last_concession_filled", "n_previous_episodes",
    "n_prior_valid_transitions", "month_sin", "month_cos",
    "chemical_historical_onset_rate",
]
EPD_FEATURE_COLUMNS = ["items_mom_growth", "total_items_log"]


def load_epd_features(path: str) -> pd.DataFrame:
    epd = pd.read_csv(path)
    epd["month"] = pd.to_datetime(epd["month"])
    epd = epd.sort_values(["chemical", "month"])
    epd["items_mom_growth"] = epd.groupby("chemical")["total_items"].pct_change()
    epd["total_items_log"] = np.log1p(epd["total_items"])
    return epd[["month", "chemical", "total_items_log", "items_mom_growth"]]


def build_full_join(features: pd.DataFrame, epd: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    df["epd_lookup_month"] = df["month"] - pd.DateOffset(months=EPD_PUBLICATION_LAG_MONTHS)
    merged = df.merge(
        epd, left_on=["epd_lookup_month", "chemical"], right_on=["month", "chemical"],
        how="left", suffixes=("", "_epd"),
    )
    return merged


def prepare_frame(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged[merged["label_onset_next_month"].notna()].copy()
    df["time_since_last_concession_filled"] = df["time_since_last_concession"].fillna(999)
    df["month_sin"] = np.sin(2 * np.pi * df["month_of_year"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_of_year"] / 12)
    df = df[df["total_items_log"].notna() & df["items_mom_growth"].notna()].copy()
    return df


def run_walk_forward(df: pd.DataFrame, feature_cols: list, label: str) -> pd.DataFrame:
    months = sorted(df["month"].unique())
    test_months = months[BURN_IN_MONTHS:-1]

    results = []
    for m in test_months:
        train = df[df["month"] < m].copy()
        test = df[df["month"] == m].copy()
        if len(train) < 50 or len(test) == 0:
            continue

        train_overall_rate = train["label_onset_next_month"].mean()
        train_hist = train["chemical_historical_onset_rate"].fillna(train_overall_rate)
        test_hist = test["chemical_historical_onset_rate"].fillna(train_overall_rate)

        X_train = train[feature_cols].copy()
        X_train["chemical_historical_onset_rate"] = train_hist
        y_train = train["label_onset_next_month"].astype(int)

        X_test = test[feature_cols].copy()
        X_test["chemical_historical_onset_rate"] = test_hist

        xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                             eval_metric="aucpr", random_state=42)
        xgb.fit(X_train, y_train)
        pred = xgb.predict_proba(X_test)[:, 1]

        results.append(pd.DataFrame({
            "month": m, "chemical": test["chemical"].values,
            "y_true": test["label_onset_next_month"].astype(int).values,
            "pred": pred,
        }))

    r = pd.concat(results, ignore_index=True)
    pr_auc = average_precision_score(r["y_true"], r["pred"])
    rng = np.random.default_rng(42)
    p10 = r.groupby("month").apply(
        lambda g: precision_at_k(g["y_true"].values, g["pred"].values, k=10, rng=rng),
        include_groups=False,
    ).mean()
    print(f"=== {label} ===")
    print(f"  n_test_months={r['month'].nunique()}  n_rows={len(r)}  n_positives={r['y_true'].sum()}")
    print(f"  base_rate={r['y_true'].mean():.4f}")
    print(f"  PR-AUC={pr_auc:.4f}")
    print(f"  precision@10={p10:.4f}")
    return r


if __name__ == "__main__":
    features = pd.read_parquet("data/interim/chemical_features.parquet")
    epd = load_epd_features("data/interim/chemical_prescribing_volume.csv")

    merged = build_full_join(features, epd)
    df = prepare_frame(merged)

    print(f"Fair comparison set: {df['chemical'].nunique()} chemicals, "
          f"{df['month'].nunique()} months, {len(df)} at-risk rows, "
          f"{int(df['label_onset_next_month'].sum())} onset events")
    print(f"(this is the REAL sample size the comparison below is computed on)\n")

    history_results = run_walk_forward(df, HISTORY_FEATURE_COLUMNS, "History-only (fair subset)")
    print()
    full_results = run_walk_forward(df, HISTORY_FEATURE_COLUMNS + EPD_FEATURE_COLUMNS, "History + prescribing features")

    history_results.to_parquet("data/interim/phase2_history_only_predictions.parquet")
    full_results.to_parquet("data/interim/phase2_history_plus_epd_predictions.parquet")
    print("\nSaved both prediction sets to data/interim/")