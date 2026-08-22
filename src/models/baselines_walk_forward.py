"""
Phase 1 baselines + walk-forward evaluation, per the plan:

  1. Overall base rate    -- historical onset rate, computed from training data only
  2. Seasonal base rate   -- onset rate conditioned on month-of-year, training data only
  3. Persistence          -- diagnostic only (already reported in build_chemical_panel.py,
                              82.4%); not a candidate for the onset task itself, since it
                              answers a different question (stays-on rate, not onset rate)
  4. History-only logistic -- the first real model, using build_history_features.py's output

WALK-FORWARD, NOT RANDOM SPLIT (this is not optional):
Each cutoff month M: train on every valid-labeled row with month < M (expanding
window, all history up to but not including M), test on rows with month == M
(predicting onset into M+1). Never train on a month and test on an earlier one --
a shuffled random split would let the model see 2026 data while "predicting"
2021, which isn't a forecasting problem at all, and would flatter every model
here by an amount that has nothing to do with real forecasting skill.

BURN-IN: the first 12 months are used for training only, never as a test month --
chemical_historical_onset_rate is noisy or undefined with very little prior history,
and evaluating on those months would be judging the model on cases where its own
features barely exist yet.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

BURN_IN_MONTHS = 12


def prepare_model_frame(features: pd.DataFrame) -> pd.DataFrame:
    """At-risk rows only (clear this month, with a defined next-month label),
    with model-ready feature columns (no NaNs left for the logistic model)."""
    df = features[features["label_onset_next_month"].notna()].copy()

    # chemical_historical_onset_rate is NaN for chemicals with zero prior
    # transitions -- fill with the training-set overall rate at prediction
    # time, not a single global constant computed from the whole dataset
    # (that would itself leak future information into early predictions).
    # Handled inside the walk-forward loop, not here -- flagging it so it's
    # not missed.

    # time_since_last_concession is NaN for chemicals never seen on
    # concession at all in our data window -- a large sentinel value
    # communicates "very long / unknown" to the model without an NaN error.
    df["time_since_last_concession_filled"] = df["time_since_last_concession"].fillna(999)

    # cyclic encoding for month-of-year so December and January are close
    # to each other numerically, not 11 apart
    df["month_sin"] = np.sin(2 * np.pi * df["month_of_year"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_of_year"] / 12)

    return df


FEATURE_COLUMNS = [
    "streak_length",
    "time_since_last_concession_filled",
    "n_previous_episodes",
    "n_prior_valid_transitions",
    "month_sin",
    "month_cos",
]


def walk_forward_evaluate(df: pd.DataFrame) -> pd.DataFrame:
    months = sorted(df["month"].unique())
    test_months = months[BURN_IN_MONTHS:-1]  # last month has no next-month label to test against

    results = []
    for m in test_months:
        train = df[df["month"] < m].copy()
        test = df[df["month"] == m].copy()
        if len(train) < 50 or len(test) == 0:
            continue  # too little history to fit anything meaningful yet

        # chemical_historical_onset_rate: fill missing values with THIS
        # TRAINING WINDOW's overall rate, not a global constant -- keeps
        # the walk-forward split honest at every step
        train_overall_rate = train["label_onset_next_month"].mean()
        train_hist_rate_filled = train["chemical_historical_onset_rate"].fillna(train_overall_rate)
        test_hist_rate_filled = test["chemical_historical_onset_rate"].fillna(train_overall_rate)

        # ---- Baseline 1: overall base rate ----
        pred_overall = pd.Series(train_overall_rate, index=test.index)

        # ---- Baseline 2: seasonal base rate ----
        seasonal_rates = train.groupby("month_of_year")["label_onset_next_month"].mean()
        pred_seasonal = test["month_of_year"].map(seasonal_rates).fillna(train_overall_rate)

        # ---- Model: history-only logistic regression ----
        X_train = train[FEATURE_COLUMNS].copy()
        X_train["chemical_historical_onset_rate"] = train_hist_rate_filled
        y_train = train["label_onset_next_month"].astype(int)

        X_test = test[FEATURE_COLUMNS].copy()
        X_test["chemical_historical_onset_rate"] = test_hist_rate_filled

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        pred_logistic = clf.predict_proba(X_test)[:, 1]

        # NOTE: scale_pos_weight was tested here and REMOVED after checking
        # calibration -- it improved nothing (PR-AUC/precision@10 were
        # actually *worse* with it) and badly distorted the model's
        # probability outputs (predictions of ~0.77 corresponded to an
        # actual onset rate of ~0.11). Gradient-boosted trees generally
        # don't need class rebalancing to find rare-class signal -- their
        # splits are based on information gain, not accuracy. Rebalancing
        # here was "fixing" a problem the model didn't have, at a real cost.
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="aucpr",
            random_state=42,
        )
        xgb.fit(X_train, y_train)
        pred_xgb = xgb.predict_proba(X_test)[:, 1]

        month_result = pd.DataFrame({
            "month": m,
            "chemical": test["chemical"].values,
            "y_true": test["label_onset_next_month"].astype(int).values,
            "pred_overall_base_rate": pred_overall.values,
            "pred_seasonal_base_rate": pred_seasonal.values,
            "pred_logistic": pred_logistic,
            "pred_xgb": pred_xgb,
        })
        results.append(month_result)

    return pd.concat(results, ignore_index=True)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10, rng: np.random.Generator = None) -> float:
    """Of the top-k highest-scored chemicals, what fraction actually onset?

    Ties are broken RANDOMLY, not by np.argsort's default (stable, original-
    order) behaviour. This matters a lot here: the two baseline models
    predict a single identical value for every chemical in a given month --
    a fully-tied array. Breaking ties by original row order silently turns
    "top 10 by risk" into "whichever 10 happen to sort first", an arbitrary
    selection with no relationship to actual risk, which is a different
    (and meaningless) thing to be measuring for an uninformative baseline.
    Random tie-breaking makes a baseline with zero real signal converge, in
    expectation, to the true base rate -- which is the correct way to
    represent "no real per-item ranking exists here".
    """
    if rng is None:
        rng = np.random.default_rng()
    if len(y_true) < k:
        k = len(y_true)
    jitter = rng.uniform(-1e-9, 1e-9, size=len(y_score))
    top_k_idx = np.argsort(-(y_score + jitter))[:k]
    return y_true[top_k_idx].mean()


def summarize(results: pd.DataFrame, score_col: str, seed: int = 42) -> dict:
    pr_auc = average_precision_score(results["y_true"], results[score_col])

    rng = np.random.default_rng(seed)
    per_month_p_at_10 = results.groupby("month").apply(
        lambda g: precision_at_k(g["y_true"].values, g[score_col].values, k=10, rng=rng),
        include_groups=False,
    )

    return {
        "pr_auc": pr_auc,
        "mean_precision_at_10": per_month_p_at_10.mean(),
        "n_test_months": results["month"].nunique(),
        "n_test_rows": len(results),
        "n_positives": results["y_true"].sum(),
        "base_rate_in_test_set": results["y_true"].mean(),
    }


if __name__ == "__main__":
    features = pd.read_parquet("data/interim/chemical_features.parquet")
    df = prepare_model_frame(features)

    print(f"Running walk-forward evaluation: {df['month'].nunique()} months, "
          f"burn-in={BURN_IN_MONTHS}, {len(df)} total at-risk rows")
    print()

    results = walk_forward_evaluate(df)

    for name, col in [
        ("Overall base rate", "pred_overall_base_rate"),
        ("Seasonal base rate", "pred_seasonal_base_rate"),
        ("History-only logistic", "pred_logistic"),
        ("Gradient boosted (XGBoost)", "pred_xgb"),
    ]:
        stats = summarize(results, col)
        print(f"=== {name} ===")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print()

    results.to_parquet("data/interim/walk_forward_predictions.parquet")
    print("Saved per-row predictions to data/interim/walk_forward_predictions.parquet")

    print("\n=== Calibration check: Gradient boosted (XGBoost), the winning model ===")
    print("(the plan's exit criteria require this, not just a good precision@10 --")
    print(" a model can rank well while still being a badly wrong probability estimator)")
    frac_pos, mean_pred = calibration_curve(
        results["y_true"], results["pred_xgb"], n_bins=10, strategy="quantile"
    )
    for mp, fp in zip(mean_pred, frac_pos):
        flag = "" if abs(mp - fp) < 0.03 else "  <- notable gap"
        print(f"  predicted ~{mp:.3f}  ->  actual onset rate {fp:.3f}{flag}")