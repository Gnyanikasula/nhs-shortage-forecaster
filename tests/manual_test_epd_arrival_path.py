"""
Manual end-to-end test for the EPD-arrival path in run_monthly_update_db.py --
exercises the real Spark job, the real Postgres writes, and the real
scoring step, WITHOUT downloading a real 7+GB NHSBSA file. A tiny
synthetic EPD CSV is served over a local HTTP server and used as the
"download" URL instead.

SAFETY: refuses to run unless DATABASE_URL points at localhost/127.0.0.1.
This test writes and deletes real rows -- it must never touch Neon.

Usage:
    docker compose up -d db          # start local Postgres only
    export DATABASE_URL="postgresql://nhs_forecaster:<pw>@localhost:5432/nhs_forecaster"
    python tests/manual_test_epd_arrival_path.py
"""
import os
import sys
import threading
import http.server
import functools
import tempfile
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "db"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "spark"))

DATABASE_URL = os.environ.get("DATABASE_URL", "")
if "localhost" not in DATABASE_URL and "127.0.0.1" not in DATABASE_URL:
    print("REFUSING TO RUN: DATABASE_URL does not look like a local database.")
    print(f"  DATABASE_URL={DATABASE_URL!r}")
    print("This test writes and deletes real data -- it must only ever run")
    print("against the local docker-compose Postgres, never Neon.")
    sys.exit(1)

from schema import get_engine, init_schema, pipeline_state, epd_prescribing_features, prediction_log
from sqlalchemy import select, insert as sa_insert, delete
from rebuild_panel_from_source import rebuild_everything

import run_monthly_update_db as target


def pick_target_month() -> str:
    """Use the latest month actually present in the real concession
    archive so score_and_log() has real at-risk chemicals to score --
    avoids hardcoding a month that may age out of relevance."""
    rebuilt = rebuild_everything()
    latest = rebuilt["features"]["month"].max()
    return latest.strftime("%Y%m")


def write_synthetic_epd_csv(path: str, yyyymm: str, dash_format: bool):
    """Deliberately writes YEAR_MONTH in dash format ("2026-08") when
    dash_format=True, to specifically exercise the normalization fix
    against the mismatch suspected in real NHSBSA exports."""
    ym_value = f"{yyyymm[:4]}-{yyyymm[4:]}" if dash_format else yyyymm
    rows = [
        "YEAR_MONTH,BNF_CHEMICAL_SUBSTANCE_CODE,BNF_CHEMICAL_SUBSTANCE,PRACTICE_CODE,ICB_CODE,ITEMS",
    ]
    chemicals = [("0407010H0", "Paracetamol"), ("0212000B0", "Atorvastatin")]
    practices = ["A81001", "A81002", "A81003"]
    for code, name in chemicals:
        for i, practice in enumerate(practices):
            rows.append(f"{ym_value},{code},{name},{practice},ICB01,{100 + i * 10}")
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")


def serve_dir(directory: str, port: int):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    httpd = http.server.HTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def cleanup(engine, yyyymm: str):
    with engine.begin() as conn:
        conn.execute(delete(epd_prescribing_features).where(epd_prescribing_features.c.year_month == yyyymm))
        conn.execute(delete(prediction_log).where(prediction_log.c.month == yyyymm))
        conn.execute(delete(pipeline_state))


def main():
    engine = init_schema()
    yyyymm = pick_target_month()
    print(f"Target month for this test: {yyyymm}")

    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, f"epd_snomed_{yyyymm}.csv")
    write_synthetic_epd_csv(csv_path, yyyymm, dash_format=True)

    httpd = serve_dir(tmp_dir, 8765)
    url = f"http://127.0.0.1:8765/{os.path.basename(csv_path)}"

    try:
        cleanup(engine, yyyymm)  # clear any leftovers from a previous failed run

        with engine.begin() as conn:
            conn.execute(sa_insert(pipeline_state).values(next_month=yyyymm, url=url))

        print("--- First run (should process fully) ---")
        target.main()

        with engine.begin() as conn:
            rows = conn.execute(
                select(epd_prescribing_features).where(epd_prescribing_features.c.year_month == yyyymm)
            ).fetchall()
        assert len(rows) > 0, "No epd_prescribing_features rows written"
        for r in rows:
            assert r.year_month == yyyymm, (
                f"BUG STILL PRESENT: year_month stored as {r.year_month!r}, "
                f"expected canonical {yyyymm!r} -- normalization fix not applied "
                f"or not working."
            )
        print(f"  PASS: {len(rows)} feature rows, year_month correctly normalized to {yyyymm!r}")

        with engine.begin() as conn:
            pred_rows = conn.execute(
                select(prediction_log).where(prediction_log.c.month == yyyymm)
            ).fetchall()
        assert len(pred_rows) > 0, "No predictions logged"
        print(f"  PASS: {len(pred_rows)} predictions logged for {yyyymm}")

        with engine.begin() as conn:
            state = conn.execute(select(pipeline_state)).fetchall()
        assert all(s.next_month is None for s in state), "pipeline_state not cleared after processing"
        print("  PASS: pipeline_state cleared")

        print("--- Second run (should no-op via already_processed) ---")
        with engine.begin() as conn:
            conn.execute(sa_insert(pipeline_state).values(next_month=yyyymm, url=url))
        target.main()  # should detect already_processed and clear config without reprocessing
        with engine.begin() as conn:
            state = conn.execute(select(pipeline_state)).fetchall()
        assert all(s.next_month is None for s in state), "Second run did not clear pipeline_state"
        print("  PASS: idempotent re-run handled correctly")

        print("\nALL CHECKS PASSED")

    finally:
        httpd.shutdown()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        cleanup(engine, yyyymm)
        print("Cleaned up test rows and temp files.")


if __name__ == "__main__":
    main()