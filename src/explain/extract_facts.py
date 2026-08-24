"""
Turns a flagged chemical's raw XGBoost feature contributions into a
small, ranked set of plain-language facts -- the ONLY input the LLM
phrasing step (phrase_explanation.py) is ever given. No free text, no
external source: every fact traces directly to a specific column value
on this chemical's own row.

Uses XGBoost's built-in per-instance feature contributions
(pred_contribs=True) rather than adding the separate `shap` library --
the model already computes this natively, so pulling in a new
dependency for something already available for free would be exactly
the kind of unnecessary tech this project's own plan doc warns against.
"""
import os
import sys

import numpy as np
import xgboost as xgb

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "db"))

from rebuild_panel_from_source import rebuild_everything


def explain_chemical(row, contribs, feature_columns):
    """
    row: a pandas Series -- one at-risk chemical's raw feature values
    contribs: array from booster.predict(..., pred_contribs=True)[0],
              same order as feature_columns + ["bias"]
    Returns: list of {"fact": str, "weight": float}, ranked by |weight|,
             ready to hand to phrase_explanation.py as structured JSON.
    """
    contrib_map = dict(zip(feature_columns, contribs[:-1]))  # drop bias

    facts = []

    # streak_length and time_since_last_concession_filled both describe
    # "how recently did this chemical last leave a concession" -- treat
    # as ONE fact, keyed on whichever has the larger contribution, so we
    # never show two near-duplicate sentences.
    recency_weight = max(
        abs(contrib_map["streak_length"]),
        abs(contrib_map["time_since_last_concession_filled"]),
    )
    months_clear = int(row["streak_length"])
    if row["time_since_last_concession_filled"] >= 999:
        recency_fact = "has never been on a price concession in the archive's history"
    elif months_clear <= 1:
        recency_fact = "came off a price concession last month"
    else:
        recency_fact = f"has been clear of concession for {months_clear} consecutive months"
    facts.append({"fact": recency_fact, "weight": recency_weight})

    n_episodes = int(row["n_previous_episodes"])
    facts.append({
        "fact": f"has had {n_episodes} previous concession episode{'s' if n_episodes != 1 else ''}",
        "weight": abs(contrib_map["n_previous_episodes"]),
    })

    rate_pct = row["chemical_historical_onset_rate"] * 100
    n_obs = int(row["n_prior_valid_transitions"])
    # n_prior_valid_transitions itself is sample size, not a risk driver --
    # folded in as a confidence caveat on the rate fact, never shown alone.
    confidence_note = f" (based on {n_obs} observed months)" if n_obs > 0 else " (limited history)"
    facts.append({
        "fact": f"has historically gone onto concession {rate_pct:.0f}% of the time it's been in this position{confidence_note}",
        "weight": abs(contrib_map["chemical_historical_onset_rate"]),
    })

    # month_sin/month_cos are two numeric halves of one seasonal signal --
    # never ranked as separate features. Combined magnitude decides
    # whether "time of year" is worth mentioning at all.
    seasonal_weight = abs(contrib_map["month_sin"]) + abs(contrib_map["month_cos"])
    month_name = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ][int(row["month_of_year"]) - 1]
    direction = "higher" if contrib_map["month_sin"] + contrib_map["month_cos"] > 0 else "lower"
    facts.append({
        "fact": f"{month_name} has shown a slightly {direction} than average onset pattern historically",
        "weight": seasonal_weight,
    })

    facts.sort(key=lambda f: f["weight"], reverse=True)
    return facts[:3]


def get_contributions_for_chemical(row, feature_columns, model):
    """Returns the raw pred_contribs array for a single at-risk chemical's
    row. Shared by extract_facts's own test run and phrase_explanation.py
    so both use exactly the same mechanism."""
    booster = model.get_booster()
    # row_df = row[feature_columns].to_frame().T
        # .astype(float) matters here: a single row pulled out as a Series
    # collapses to a generic "object" dtype (a Series can only have one
    # dtype, and the row spans several originally-numeric columns), and
    # that object dtype propagates straight into the DataFrame XGBoost
    # sees -- it refuses non-numeric dtypes outright. Casting restores
    # the real numeric dtype before DMatrix construction.
    row_df = row[feature_columns].astype(float).to_frame().T
    dmatrix = xgb.DMatrix(row_df, feature_names=feature_columns)
    return booster.predict(dmatrix, pred_contribs=True)[0]


if __name__ == "__main__":
    rebuilt = rebuild_everything()
    features = rebuilt["features"]
    bundle = rebuilt["model_bundle"]
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    latest_month = features["month"].max()
    at_risk = features[(features["month"] == latest_month) & (~features["on_concession"])].copy()

    at_risk["time_since_last_concession_filled"] = at_risk["time_since_last_concession"].fillna(999)
    at_risk["month_sin"] = np.sin(2 * np.pi * at_risk["month_of_year"] / 12)
    at_risk["month_cos"] = np.cos(2 * np.pi * at_risk["month_of_year"] / 12)
    at_risk["chemical_historical_onset_rate"] = at_risk["chemical_historical_onset_rate"].fillna(
        bundle["fallback_historical_rate"]
    )

    X = at_risk[feature_columns]
    at_risk["score"] = model.predict_proba(X)[:, 1]

    top = at_risk.sort_values("score", ascending=False).iloc[0]
    print(f"Chemical: {top['chemical']}")
    print(f"Score: {top['score']:.4f}")

    contribs = get_contributions_for_chemical(top, feature_columns, model)
    facts = explain_chemical(top, contribs, feature_columns)
    print()
    print("Top facts:")
    for f in facts:
        print(f"  [{f['weight']:.4f}] {f['fact']}")