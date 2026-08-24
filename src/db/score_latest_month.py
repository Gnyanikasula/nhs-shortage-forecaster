"""
Scores the most recent month in the freshly-rebuilt panel and logs to
prediction_log -- decoupled from run_monthly_update_db.py's EPD-arrival
trigger, because the production model is history-only and doesn't need
new EPD data to produce a fresh score. Safe and cheap to run every
workflow invocation (seconds, not the 20+ minutes a real EPD month costs).

This also seeds prediction_log with real data before the dashboard/API
have anything meaningful to show -- without this, a fresh deployment
would have an empty table and nothing to display.

MLFLOW: this is the retrain path that actually fires every single day
via the GitHub Actions cron (run_monthly_update_db.py's own retrain only
fires on the rare day a new EPD month lands). Tracking is logged here,
not just there, or the daily production retrain would go completely
unrecorded. Uses the existing Neon Postgres as the MLflow tracking
store -- GitHub Actions runners are ephemeral, so a local file-based
store would lose all history between runs.

PHASE 5 EXPLANATIONS: generated here for the top 10 by score only, not
all ~250 at-risk chemicals -- matches the project's own precision@10
metric from Phase 1, and keeps Ollama calls (each up to 3 retries) to a
bounded, predictable cost per run instead of scaling with the size of
the at-risk set. Requires Ollama reachable at localhost:11434 -- see
.github/workflows/monthly_pipeline.yml for how that's started on the
runner before this script executes inside its container.
"""
import os
import sys
from datetime import datetime

import numpy as np
import mlflow
import requests
from sqlalchemy.dialects.postgresql import insert as pg_insert

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "explain"))

from schema import get_engine, init_schema, prediction_log
from rebuild_panel_from_source import rebuild_everything
from extract_facts import explain_chemical, get_contributions_for_chemical
from phrase_explanation import generate_explanation

TOP_N_EXPLAINED = 10


def _ollama_reachable() -> bool:
    """Cheap pre-flight check. Without this, a fully-down Ollama means
    burning through TOP_N_EXPLAINED x MAX_ATTEMPTS (up to 30) slow
    connection-timeout failures one at a time before the loop gives up
    on its own -- this fails fast instead, in under a second."""
    try:
        requests.get("http://localhost:11434", timeout=3)
        return True
    except requests.exceptions.RequestException:
        return False


def _generate_explanations(at_risk, feature_columns, model):
    """Isolated on purpose: CORE SCORING MUST SURVIVE THIS FAILING.
    Phase 5 (explanations) is an enhancement on top of Phase 1 (the
    proven, load-bearing forecaster) -- an Ollama outage or any bug in
    this optional layer must never prevent that day's real predictions
    from being written to prediction_log. Returns at_risk with
    explanation/explanation_method columns filled in wherever generation
    succeeded; left as None everywhere it didn't."""
    at_risk["explanation"] = None
    at_risk["explanation_method"] = None

    if not _ollama_reachable():
        print("  Ollama unreachable -- skipping explanations for this run "
              "(core scoring is unaffected).")
        return at_risk

    top_idx = at_risk.sort_values("phase1_production_score", ascending=False).head(TOP_N_EXPLAINED).index
    print(f"  Generating explanations for top {len(top_idx)} flagged chemicals...")
    for idx in top_idx:
        row = at_risk.loc[idx]
        try:
            contribs = get_contributions_for_chemical(row, feature_columns, model)
            facts = explain_chemical(row, contribs, feature_columns)
            explanation, method = generate_explanation(row["chemical"], row["phase1_production_score"], facts)
            at_risk.loc[idx, "explanation"] = explanation
            at_risk.loc[idx, "explanation_method"] = method
            print(f"    {row['chemical']}: {method}")
        except Exception as e:
            # One chemical failing (a stray Ollama timeout, an unexpected
            # data issue) must not skip the remaining nine, and must
            # never propagate up into score_latest_month() -- that would
            # take down the core upsert below along with it.
            print(f"    {row['chemical']}: FAILED ({type(e).__name__}: {e}) -- leaving unexplained, continuing.")

    return at_risk


def score_latest_month(engine) -> int:
    mlflow.set_tracking_uri(os.environ["DATABASE_URL"])
    mlflow.set_experiment("nhs-shortage-production-retrain")

    print(f"[{datetime.now()}] Rebuilding panel and scoring latest month...")
    rebuilt = rebuild_everything()
    features = rebuilt["features"]
    bundle = rebuilt["model_bundle"]
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    latest_month = features["month"].max()
    at_risk = features[(features["month"] == latest_month) & (~features["on_concession"])].copy()

    if len(at_risk) == 0:
        print(f"  No at-risk chemicals found for {latest_month} -- nothing to score.")
        return 0

    # rebuild_everything() returns RAW features -- these derived columns are
    # only created inside prepare_model_frame() during training, which never
    # touches the at-risk scoring subset. Bug caught by real testing: without
    # this, bundle["feature_columns"] references columns that don't exist yet.
    at_risk["time_since_last_concession_filled"] = at_risk["time_since_last_concession"].fillna(999)
    at_risk["month_sin"] = np.sin(2 * np.pi * at_risk["month_of_year"] / 12)
    at_risk["month_cos"] = np.cos(2 * np.pi * at_risk["month_of_year"] / 12)
    at_risk["chemical_historical_onset_rate"] = at_risk["chemical_historical_onset_rate"].fillna(
        bundle["fallback_historical_rate"]
    )

    X = at_risk[feature_columns]
    at_risk["phase1_production_score"] = model.predict_proba(X)[:, 1]
    at_risk["phase3_shadow_score"] = None

    yyyymm = latest_month.strftime("%Y%m")

    # Isolated call -- see _generate_explanations' docstring. Whatever
    # happens inside (Ollama down, a bug, partial success), execution
    # always continues to the upsert below with whatever was gathered.
    at_risk = _generate_explanations(at_risk, feature_columns, model)
    n_explained = at_risk["explanation"].notna().sum()

    log_rows = at_risk[[
        "chemical", "phase1_production_score", "phase3_shadow_score",
        "explanation", "explanation_method",
    ]].copy()
    log_rows["month"] = yyyymm

    with engine.begin() as conn:
        stmt = pg_insert(prediction_log).values(log_rows.to_dict(orient="records"))
        stmt = stmt.on_conflict_do_update(
            index_elements=["chemical", "month"],
            set_={
                "phase1_production_score": stmt.excluded.phase1_production_score,
                "phase3_shadow_score": stmt.excluded.phase3_shadow_score,
                "explanation": stmt.excluded.explanation,
                "explanation_method": stmt.excluded.explanation_method,
                "scored_at": stmt.excluded.scored_at,
            },
        )
        conn.execute(stmt)

    print(f"  Scored and logged {len(log_rows)} chemicals for {yyyymm} ({n_explained} with explanations).")

    with mlflow.start_run(run_name=f"daily_score_{yyyymm}"):
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "feature_columns": ",".join(feature_columns),
        })
        mlflow.log_metrics({
            "training_rows": int(features["label_onset_next_month"].notna().sum()),
            "onset_events": int(features["label_onset_next_month"].sum()),
            "fallback_historical_rate": float(bundle["fallback_historical_rate"]),
            "at_risk_scored": len(at_risk),
            "explanations_generated": int(n_explained),
        })
    print(f"  Logged retrain metadata to MLflow (experiment: nhs-shortage-production-retrain).")

    return len(log_rows)


if __name__ == "__main__":
    engine = init_schema()
    n = score_latest_month(engine)
    print(f"Done. {n} predictions logged.")