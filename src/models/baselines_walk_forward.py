"""
Phase 1 baselines + walk-forward evaluation, per the plan.

WALK-FORWARD, NOT RANDOM SPLIT: each cutoff month M trains on every valid
row with month < M, tests on month == M (predicting onset into M+1). A
shuffled split would let the model see 2026 data while "predicting" 2021.

BURN-IN: first 12 months train-only, never tested -- history features are
too thin that early to fairly judge the model on them.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

BURN_IN_MONTHS = 12


def prepare_model_frame(features: pd.DataFrame) -> pd.DataFrame:
    df = features[features["label_onset_next_month"].notna()].copy()
    df["time_since_last_concession_filled"] = df["time_since_last_concession"].fillna(999)
    df["month_sin"] = np.sin(2 * np.pi * df["month_of_year"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_of_year"] / 12)
    return df


FEATURE_COLUMNS = [
    "streak_length", "time_since_last_concession_filled", "n_previous_episodes",
    "n_prior_valid_transitions", "month_sin", "month_cos",
]


def walk_forward_evaluate(df: pd.DataFrame) -> pd.DataFrame:
    months = sorted(df["month"].unique())
    test_months = months[BURN_IN_MONTHS:-1]

    results = []
    for m in test_months:
        train = df[df["month"] < m].copy()
        test = df[df["month"] == m].copy()
        if len(train) < 50 or len(test) == 0:
            continue

        train_overall_rate = train["label_onset_next_month"].mean()
        train_hist_rate_filled = train["chemical_historical_onset_rate"].fillna(train_overall_rate)
        test_hist_rate_filled = test["chemical_historical_onset_rate"].fillna(train_overall_rate)

        pred_overall = pd.Series(train_overall_rate, index=test.index)

        seasonal_rates = train.groupby("month_of_year")["label_onset_next_month"].mean()
        pred_seasonal = test["month_of_year"].map(seasonal_rates).fillna(train_overall_rate)

        X_train = train[FEATURE_COLUMNS].copy()
        X_train["chemical_historical_onset_rate"] = train_hist_rate_filled
        y_train = train["label_onset_next_month"].astype(int)

        X_test = test[FEATURE_COLUMNS].copy()
        X_test["chemical_historical_onset_rate"] = test_hist_rate_filled

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)
        pred_logistic = clf.predict_proba(X_test)[:, 1]

        month_result = pd.DataFrame({
            "month": m,
            "chemical": test["chemical"].values,
            "y_true": test["label_onset_next_month"].astype(int).values,
            "pred_overall_base_rate": pred_overall.values,
            "pred_seasonal_base_rate": pred_seasonal.values,
            "pred_logistic": pred_logistic,
        })
        results.append(month_result)

    return pd.concat(results, ignore_index=True)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10, rng: np.random.Generator = None) -> float:
    """Ties broken RANDOMLY -- critical here, since the baselines predict a
    single identical value for every chemical in a month. Breaking ties by
    original row order would silently measure 'whichever 10 sort first',
    not 'top 10 by risk'. Random tie-breaking makes an uninformative
    baseline converge, in expectation, to the true base rate."""
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
    ]:
        stats = summarize(results, col)
        print(f"=== {name} ===")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print()

    results.to_parquet("data/interim/walk_forward_predictions.parquet")
    print("Saved per-row predictions to data/interim/walk_forward_predictions.parquet")