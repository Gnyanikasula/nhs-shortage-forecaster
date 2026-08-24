"""
FastAPI endpoint serving the current risk predictions from Postgres.
Read-only -- this app never writes to the database, only the pipeline
scripts do. Deliberately thin: no business logic here, just querying
and shaping what's already been computed and logged.
"""
import os
import sys
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import select, func

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "db"))

from schema import get_engine, prediction_log, actual_outcomes

app = FastAPI(
    title="NHS Drug Shortage Early Warning API",
    description="Read-only API serving monthly onset-risk predictions for NHS price concessions.",
    version="1.0",
)

engine = get_engine()


@app.get("/health")
def health():
    """Confirms the API is up AND can reach the database -- two
    different failure modes, worth distinguishing in the response."""
    try:
        with engine.connect() as conn:
            conn.execute(select(func.now()))
        return {"status": "ok", "database": "reachable", "checked_at": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unreachable: {e}")


@app.get("/predictions/latest")
def latest_predictions(limit: int = Query(default=20, ge=1, le=377)):
    """Returns the most recent month's predictions, ranked by risk score
    descending. limit is capped at 377 (the full chemical universe) --
    asking for more than that would silently return everything anyway,
    better to be explicit about the real ceiling."""
    with engine.connect() as conn:
        latest_month_row = conn.execute(
            select(func.max(prediction_log.c.month))
        ).scalar()

        if latest_month_row is None:
            return {"month": None, "predictions": [], "note": "No predictions logged yet."}

        rows = conn.execute(
            select(prediction_log)
            .where(prediction_log.c.month == latest_month_row)
            .order_by(prediction_log.c.phase1_production_score.desc())
            .limit(limit)
        ).fetchall()

    return {
        "month": latest_month_row,
        "n_returned": len(rows),
        "predictions": [
            {
                "chemical": r.chemical,
                "phase1_production_score": round(r.phase1_production_score, 4),
                "phase3_shadow_score": r.phase3_shadow_score,
                "scored_at": r.scored_at.isoformat() if r.scored_at else None,
            }
            for r in rows
        ],
    }


@app.get("/outcomes/{chemical}")
def chemical_history(chemical: str, months: int = Query(default=12, ge=1, le=79)):
    """Returns a chemical's real concession history -- ground truth,
    not predictions. Useful for sanity-checking a prediction against
    what actually happened."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(actual_outcomes)
            .where(actual_outcomes.c.chemical == chemical)
            .order_by(actual_outcomes.c.month.desc())
            .limit(months)
        ).fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No outcome history found for chemical '{chemical}'")

    return {
        "chemical": chemical,
        "history": [
            {"month": r.month, "on_concession": r.on_concession}
            for r in sorted(rows, key=lambda r: r.month)
        ],
    }


@app.get("/stats")
def stats():
    """Basic counts -- a cheap way to confirm the pipeline is actually
    producing data, without needing to inspect the database directly."""
    with engine.connect() as conn:
        n_outcomes = conn.execute(select(func.count()).select_from(actual_outcomes)).scalar()
        n_predictions = conn.execute(select(func.count()).select_from(prediction_log)).scalar()
        latest_prediction_month = conn.execute(select(func.max(prediction_log.c.month))).scalar()
        latest_outcome_month = conn.execute(select(func.max(actual_outcomes.c.month))).scalar()

    return {
        "total_outcome_rows": n_outcomes,
        "total_prediction_rows": n_predictions,
        "latest_prediction_month": latest_prediction_month,
        "latest_outcome_month": latest_outcome_month,
    }