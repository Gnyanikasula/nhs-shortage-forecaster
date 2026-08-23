"""
Phase 3: joins the Spark-derived row-level features (total_items,
n_distinct_practices, hhi) onto the Phase 1 panel and runs the same
fair, like-for-like walk-forward comparison used in Phase 2.

RESTRICTED TO UNAMBIGUOUS CHEMICALS (single resolved BNF code, ~226 of
377). For the ~133 chemicals with multiple candidate BNF codes (e.g.
Hydrocortisone across 14 sections), summing total_items across codes is
valid (volumes add), but summing n_distinct_practices or hhi is NOT --
a practice prescribing two forms of the same chemical would be double-
counted, and HHI values from different codes can't be meaningfully
combined by summing. Rather than introduce that error, or spend further
engineering re-running the Spark job grouped by concession-chemical
instead of BNF code, this test is honestly scoped to the chemicals where
the join is mathematically clean.

COVERAGE CAVEAT, STATED UP FRONT: only 6 months of raw EPD were
processed (Mar-Aug 2025), and with a 2-month publication lag, the
testable "as of" window is narrower still -- expect far fewer test
months and onset events than Phase 2's 910-event comparison. Any result
here should be read with that in mind, the same way the original
3-chemical ADHD pilot was.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

sys.path.insert(0, "src/models")
from baselines_walk_forward import precision_at_k, BURN_IN_MONTHS

EPD_PUBLICATION_LAG_MONTHS = 2
PHASE3_BURN_IN_MONTHS = 2  # NOT the imported BURN_IN_MONTHS=12 -- that's tuned for the 79-month panel, not this 6-month window

HISTORY_FEATURE_COLUMNS = [
    "streak_length", "time_since_last_concession_filled", "n_previous_episodes",
    "n_prior_valid_transitions", "month_sin", "month_cos",
    "chemical_historical_onset_rate",
]
NEW_FEATURE_COLUMNS = ["total_items_log", "n_distinct_practices_log", "hhi"]


def get_unambiguous_chemical_codes(mapping_path: str) -> dict:
    mapping = pd.read_csv(mapping_path)
    clean = mapping[mapping["status"] == "ok (prefix match)"]
    return dict(zip(clean["chemical"], clean["bnf_code"]))


def load_spark_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"])
    df["total_items_log"] = np.log1p(df["total_items"])
    df["n_distinct_practices_log"] = np.log1p(df["n_distinct_practices"])
    return df[["month", "BNF_CHEMICAL_SUBSTANCE_CODE", "total_items_log", "n_distinct_practices_log", "hhi"]]


def build_join(features: pd.DataFrame, spark_features: pd.DataFrame, code_map: dict) -> pd.DataFrame:
    df = features.copy()
    df["bnf_code"] = df["chemical"].map(code_map)
    df = df[df["bnf_code"].notna()].copy()
    print(f"Restricted to {df['chemical'].nunique()} unambiguously-mapped chemicals "
          f"({len(code_map)} available in the mapping file)")

    df["epd_lookup_month"] = df["month"] - pd.DateOffset(months=EPD_PUBLICATION_LAG_MONTHS)
    merged = df.merge(
        spark_features,
        left_on=["epd_lookup_month", "bnf_code"],
        right_on=["month", "BNF_CHEMICAL_SUBSTANCE_CODE"],
        how="left", suffixes=("", "_spark"),
    )
    return merged


def prepare_frame(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged[merged["label_onset_next_month"].notna()].copy()
    df["time_since_last_concession_filled"] = df["time_since_last_concession"].fillna(999)
    df["month_sin"] = np.sin(2 * np.pi * df["month_of_year"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_of_year"] / 12)
    df = df[df["total_items_log"].notna() & df["n_distinct_practices_log"].notna() & df["hhi"].notna()].copy()
    return df


def run_walk_forward(df: pd.DataFrame, feature_cols: list, label: str):
    # months = sorted(df["month"].unique())
    # if len(months) <= BURN_IN_MONTHS + 1:
    #     print(f"=== {label} ===")
    #     print(f"  Only {len(months)} months available with burn-in={BURN_IN_MONTHS} -- "
    #           f"not enough to run a walk-forward split. Reduce BURN_IN_MONTHS for this "
    #           f"test or gather more months before trusting any result here.")
    #     return None

    # test_months = months[BURN_IN_MONTHS:-1]
    months = sorted(df["month"].unique())
    if len(months) <= PHASE3_BURN_IN_MONTHS:
        print(f"=== {label} ===")
        print(f"  Only {len(months)} months available with burn-in={PHASE3_BURN_IN_MONTHS} -- "
              f"not enough to run any test month. Gather more months before trusting any result here.")
        return None

    # NOTE: no "-1" here, unlike baselines_walk_forward.py -- prepare_frame() already
    # dropped rows with no valid label, so every month remaining genuinely has one.
    # The "-1" in the 79-month version exists because THAT script never pre-filters --
    # copying it here would wrongly discard an already-valid final test month.
    test_months = months[PHASE3_BURN_IN_MONTHS:]
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

    if not results:
        print(f"=== {label} ===\n  No valid test months produced results.")
        return None

    r = pd.concat(results, ignore_index=True)
    pr_auc = average_precision_score(r["y_true"], r["pred"])
    rng = np.random.default_rng(42)
    p10 = r.groupby("month").apply(
        lambda g: precision_at_k(g["y_true"].values, g["pred"].values, k=min(10, len(g)), rng=rng),
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
    spark_features = load_spark_features("data/interim/epd_prescribing_features_combined.csv")
    code_map = get_unambiguous_chemical_codes("data/interim/chemical_to_bnf_code.csv")

    merged = build_join(features, spark_features, code_map)
    df = prepare_frame(merged)

    print(f"\nFair comparison set: {df['chemical'].nunique()} chemicals, "
          f"{df['month'].nunique()} months, {len(df)} at-risk rows, "
          f"{int(df['label_onset_next_month'].sum())} onset events")
    print("(this is the REAL sample size -- expect this to be small, see caveat in module docstring)\n")

    history_results = run_walk_forward(df, HISTORY_FEATURE_COLUMNS, "History-only (fair subset)")
    print()
    full_results = run_walk_forward(df, HISTORY_FEATURE_COLUMNS + NEW_FEATURE_COLUMNS, "History + Spark row-level features")

    # if history_results is not None:
    #     history_results.to_parquet("data/interim/phase3_history_only_predictions.parquet")
    # if full_results is not None:
    #     full_results.to_parquet("data/interim/phase3_history_plus_spark_predictions.parquet")
    # print("\nSaved prediction sets to data/interim/ (where results were produced)")
    
    saved_any = False
    if history_results is not None:
        history_results.to_parquet("data/interim/phase3_history_only_predictions.parquet")
        saved_any = True
    if full_results is not None:
        full_results.to_parquet("data/interim/phase3_history_plus_spark_predictions.parquet")
        saved_any = True

    if saved_any:
        print("\nSaved prediction sets to data/interim/")
    else:
        print("\nNothing saved -- neither comparison produced results. See messages above.")