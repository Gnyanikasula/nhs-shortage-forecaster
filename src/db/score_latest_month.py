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
"""
import os
import sys
from datetime import datetime

import numpy as np
from sqlalchemy.dialects.postgresql import insert as pg_insert

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from schema import get_engine, init_schema, prediction_log
from rebuild_panel_from_source import rebuild_everything


def score_latest_month(engine) -> int:
    import mlflow

    mlflow.set_tracking_uri(os.environ["DATABASE_URL"])
    mlflow.set_experiment("nhs-shortage-production-retrain")

    print(f"[{datetime.now()}] Rebuilding panel and scoring latest month...")
    rebuilt = rebuild_everything()
    features = rebuilt["features"]
    bundle = rebuilt["model_bundle"]

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

    X = at_risk[bundle["feature_columns"]]
    at_risk["phase1_production_score"] = bundle["model"].predict_proba(X)[:, 1]
    at_risk["phase3_shadow_score"] = None

    yyyymm = latest_month.strftime("%Y%m")
    log_rows = at_risk[["chemical", "phase1_production_score", "phase3_shadow_score"]].copy()
    log_rows["month"] = yyyymm

    with engine.begin() as conn:
        stmt = pg_insert(prediction_log).values(log_rows.to_dict(orient="records"))
        stmt = stmt.on_conflict_do_update(
            index_elements=["chemical", "month"],
            set_={
                "phase1_production_score": stmt.excluded.phase1_production_score,
                "phase3_shadow_score": stmt.excluded.phase3_shadow_score,
                "scored_at": stmt.excluded.scored_at,
            },
        )
        conn.execute(stmt)

    print(f"  Scored and logged {len(log_rows)} chemicals for {yyyymm}.")

    with mlflow.start_run(run_name=f"daily_score_{yyyymm}"):
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "feature_columns": ",".join(bundle["feature_columns"]),
        })
        mlflow.log_metrics({
            "training_rows": int(features["label_onset_next_month"].notna().sum()),
            "onset_events": int(features["label_onset_next_month"].sum()),
            "fallback_historical_rate": float(bundle["fallback_historical_rate"]),
            "at_risk_scored": len(at_risk),
        })
    print(f"  Logged retrain metadata to MLflow (experiment: nhs-shortage-production-retrain).")

    return len(log_rows)


if __name__ == "__main__":
    engine = init_schema()
    n = score_latest_month(engine)
    print(f"Done. {n} predictions logged.")