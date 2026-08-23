"""
Database-backed monthly orchestrator: pipeline state (what month is
next, what predictions have been logged) lives in Postgres, not a local
JSON file and CSV. This is what makes the pipeline genuinely portable to
GitHub Actions later -- an ephemeral runner has nowhere to keep a local
file between runs, but a database survives independently of any single
run.

SAFE TO SCHEDULE DAILY: on any day where pipeline_state has no row, or
its url column is empty, this exits immediately and does nothing. Once a
human writes a new (next_month, url) row via set_next_month.py, the next
scheduled run picks it up automatically.

Panel/model state is NOT persisted between runs -- score_and_log()
rebuilds the entire concession panel and retrains the production model
from source every run (see rebuild_panel_from_source.py). That is cheap
(seconds, ~9000 rows) and matches this project's standing rule:
regenerate from source rather than persist what's cheap to rebuild.
"""
import os
import sys
from datetime import datetime

import pandas as pd
from sqlalchemy import select, insert, update

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "db"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "spark"))

from schema import get_engine, init_schema, pipeline_state, prediction_log, epd_prescribing_features
from rebuild_panel_from_source import rebuild_everything


def get_current_config(engine) -> dict:
    with engine.begin() as conn:
        row = conn.execute(select(pipeline_state).order_by(pipeline_state.c.id.desc())).fetchone()
    if row is None:
        return {"next_month": None, "url": None}
    return dict(row._mapping)


def clear_config(engine, row_id: int):
    with engine.begin() as conn:
        conn.execute(update(pipeline_state).where(pipeline_state.c.id == row_id).values(next_month=None, url=None))


def already_processed(engine, yyyymm: str) -> bool:
    with engine.begin() as conn:
        row = conn.execute(
            select(epd_prescribing_features).where(epd_prescribing_features.c.year_month == yyyymm).limit(1)
        ).fetchone()
    return row is not None


def run_pipeline_for_month(engine, yyyymm: str, url: str):
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    import requests

    raw_dir = os.path.join(REPO_ROOT, "data", "raw", "epd_bulk")
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, f"epd_snomed_{yyyymm}.csv")

    print(f"[{datetime.now()}] Downloading {yyyymm}...")
    with requests.get(url, headers={"User-Agent": "student-portfolio-project"}, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(raw_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
    print(f"  Downloaded {os.path.getsize(raw_path) / 1e9:.2f} GB")

    spark_tmp_dir = os.path.join(REPO_ROOT, "data", "spark_tmp")
    spark = (
        SparkSession.builder.appName("monthly_update").master("local[4]")
        .config("spark.driver.memory", "5g")
        .config("spark.local.dir", os.path.abspath(spark_tmp_dir))
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    try:
        raw_df = spark.read.option("header", "true").csv(raw_path)
        raw_count = raw_df.count()

        df = raw_df.select(
            "YEAR_MONTH", "BNF_CHEMICAL_SUBSTANCE_CODE", "BNF_CHEMICAL_SUBSTANCE",
            "PRACTICE_CODE", "ICB_CODE", F.col("ITEMS").cast("double").alias("ITEMS"),
        ).coalesce(8)

        parquet_path = os.path.join(REPO_ROOT, "data", "interim", "epd_parquet", f"month={yyyymm}")
        df.write.mode("overwrite").parquet(parquet_path)
        check_df = spark.read.parquet(parquet_path)
        parquet_count = check_df.count()

        if raw_count != parquet_count:
            raise RuntimeError(
                f"Row count mismatch for {yyyymm}: raw={raw_count} parquet={parquet_count} -- "
                f"NOT deleting raw file, stopping for manual investigation."
            )

        practice_level = check_df.groupBy(
            "BNF_CHEMICAL_SUBSTANCE_CODE", "BNF_CHEMICAL_SUBSTANCE", "YEAR_MONTH", "PRACTICE_CODE"
        ).agg(F.sum("ITEMS").alias("practice_items"))

        chemical_level = practice_level.groupBy(
            "BNF_CHEMICAL_SUBSTANCE_CODE", "BNF_CHEMICAL_SUBSTANCE", "YEAR_MONTH"
        ).agg(
            F.sum("practice_items").alias("total_items"),
            F.count("PRACTICE_CODE").alias("n_distinct_practices"),
            F.sum(F.pow(F.col("practice_items"), 2)).alias("sum_sq"),
        ).withColumn(
            "hhi", F.col("sum_sq") / (F.col("total_items") * F.col("total_items"))
        ).drop("sum_sq")

        result_pdf = chemical_level.toPandas()
        result_pdf = result_pdf.rename(columns={
            "BNF_CHEMICAL_SUBSTANCE_CODE": "bnf_chemical_substance_code",
            "BNF_CHEMICAL_SUBSTANCE": "bnf_chemical_substance",
            "YEAR_MONTH": "year_month",
        })
        with engine.begin() as conn:
            conn.execute(insert(epd_prescribing_features), result_pdf.to_dict(orient="records"))
        print(f"  Inserted {len(result_pdf)} chemical-month feature rows into Postgres.")

        os.remove(raw_path)
        print(f"  Verified, deleted raw CSV.")
    finally:
        spark.stop()


def score_and_log(engine, yyyymm: str):
    print(f"  Rebuilding panel and model from source...")
    rebuilt = rebuild_everything()
    features = rebuilt["features"]
    bundle = rebuilt["model_bundle"]

    month_ts = pd.to_datetime(yyyymm, format="%Y%m")
    at_risk = features[(features["month"] == month_ts) & (~features["on_concession"])].copy()

    if len(at_risk) == 0:
        print(f"  No at-risk chemicals found for {yyyymm} in the freshly-rebuilt panel -- "
              f"does the concession archive actually cover this month yet?")
        return

    X = at_risk[bundle["feature_columns"]]
    at_risk["phase1_production_score"] = bundle["model"].predict_proba(X)[:, 1]
    at_risk["phase3_shadow_score"] = None

    log_rows = at_risk[["chemical", "phase1_production_score", "phase3_shadow_score"]].copy()
    log_rows["month"] = yyyymm

    with engine.begin() as conn:
        conn.execute(insert(prediction_log), log_rows.to_dict(orient="records"))
    print(f"  Logged {len(log_rows)} predictions for {yyyymm} to Postgres.")


def main():
    engine = init_schema()
    config = get_current_config(engine)

    if not config.get("url") or not config.get("next_month"):
        print(f"[{datetime.now()}] No new month configured -- nothing to do. "
              f"(This is the expected state most days.)")
        sys.exit(0)

    yyyymm = config["next_month"]
    url = config["url"]

    if already_processed(engine, yyyymm):
        print(f"[{datetime.now()}] {yyyymm} already has feature data in Postgres -- clearing config, nothing to do.")
        clear_config(engine, config["id"])
        sys.exit(0)

    print(f"[{datetime.now()}] New month configured: {yyyymm}. Running full pipeline...")
    run_pipeline_for_month(engine, yyyymm, url)
    score_and_log(engine, yyyymm)
    clear_config(engine, config["id"])
    print(f"[{datetime.now()}] Done with {yyyymm}. Config cleared -- waiting for next month's URL.")


if __name__ == "__main__":
    main()