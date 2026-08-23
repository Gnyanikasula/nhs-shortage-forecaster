"""
Semi-automated monthly pipeline. SAFE TO SCHEDULE DAILY: on any day where
the config file doesn't have a new URL waiting, this exits immediately
and does nothing -- no error, no partial work, no risk. Once a human
pastes a new month's URL into pipeline_config.json (a ~60 second task:
check NHSBSA's dataset page, copy the download link), the very next
scheduled run picks it up and does everything else automatically:
download, land as Parquet, verify, delete raw, compute Spark features,
score the Phase 1 production model against the new month, and log
predictions with a timestamp.

This is the correct design given a real, observed risk: NHSBSA changed
their download mechanism (plain CSV link -> signed, time-limited
Cloudflare URL) between one part of this project and another, with no
warning. Fully automating URL discovery against a source that can change
shape without notice is open-ended maintenance; this design instead
automates the expensive, error-prone 90% of the work and keeps the
fragile 10% (finding the correct link) as a deliberate human checkpoint.

Config file format (data/pipeline_config.json):
    {"next_month": "202609", "url": "https://.../epd_snomed_202609.csv"}
Leave "url" as null/empty until you've actually found and pasted the
real link -- an empty url means "not ready yet", not an error.
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, "src/spark")
sys.path.insert(0, "src/models")

CONFIG_PATH = "data/pipeline_config.json"
LOG_PATH = "data/interim/prediction_log.csv"


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        default = {"next_month": None, "url": None}
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(default, f, indent=2)
        return default
    with open(CONFIG_PATH) as f:
        return json.load(f)


def clear_config():
    with open(CONFIG_PATH, "w") as f:
        json.dump({"next_month": None, "url": None}, f, indent=2)


def already_processed(yyyymm: str) -> bool:
    parquet_path = f"data/interim/epd_parquet/month={yyyymm}"
    return os.path.exists(parquet_path)


def run_pipeline_for_month(yyyymm: str, url: str):
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    import requests

    raw_dir = "data/raw/epd_bulk"
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, f"epd_snomed_{yyyymm}.csv")

    print(f"[{datetime.now()}] Downloading {yyyymm}...")
    with requests.get(url, headers={"User-Agent": "student-portfolio-project"}, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(raw_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
    print(f"  Downloaded {os.path.getsize(raw_path) / 1e9:.2f} GB")

    spark = (
        SparkSession.builder.appName("monthly_update").master("local[4]")
        .config("spark.driver.memory", "5g")
        .config("spark.local.dir", os.path.abspath("data/spark_tmp"))
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

        parquet_path = f"data/interim/epd_parquet/month={yyyymm}"
        df.write.mode("overwrite").parquet(parquet_path)
        check_df = spark.read.parquet(parquet_path)
        parquet_count = check_df.count()

        if raw_count != parquet_count:
            raise RuntimeError(f"Row count mismatch for {yyyymm}: raw={raw_count} parquet={parquet_count} -- "
                                f"NOT deleting raw file, stopping for manual investigation.")

        practice_level = check_df.groupBy(
            "BNF_CHEMICAL_SUBSTANCE_CODE", "BNF_CHEMICAL_SUBSTANCE", "YEAR_MONTH", "PRACTICE_CODE"
        ).agg(F.sum("ITEMS").alias("practice_items"))
        chemical_level = practice_level.groupBy(
            "BNF_CHEMICAL_SUBSTANCE_CODE", "BNF_CHEMICAL_SUBSTANCE", "YEAR_MONTH"
        ).agg(
            F.sum("practice_items").alias("total_items"),
            F.count("PRACTICE_CODE").alias("n_distinct_practices"),
            F.sum(F.pow(F.col("practice_items"), 2)).alias("sum_sq"),
        ).withColumn("hhi", F.col("sum_sq") / (F.col("total_items") * F.col("total_items"))).drop("sum_sq")

        features_dir = "data/interim/epd_prescribing_features"
        os.makedirs(features_dir, exist_ok=True)
        chemical_level.toPandas().to_csv(f"{features_dir}/features_{yyyymm}.csv", index=False)

        os.remove(raw_path)
        print(f"  Verified, saved features, deleted raw CSV.")
    finally:
        spark.stop()


def score_and_log(yyyymm: str):
    features = pd.read_parquet("data/interim/chemical_features.parquet")
    month_ts = pd.to_datetime(yyyymm, format="%Y%m")
    at_risk = features[(features["month"] == month_ts) & (~features["on_concession"])].copy()

    if len(at_risk) == 0:
        print(f"  No at-risk chemicals found for {yyyymm} in the panel -- "
              f"has the Phase 1 panel been rebuilt to include this month?")
        return

    at_risk["time_since_last_concession_filled"] = at_risk["time_since_last_concession"].fillna(999)
    at_risk["month_sin"] = np.sin(2 * np.pi * at_risk["month_of_year"] / 12)
    at_risk["month_cos"] = np.cos(2 * np.pi * at_risk["month_of_year"] / 12)

    bundle = joblib.load("data/interim/phase1_production_model.joblib")
    at_risk["chemical_historical_onset_rate"] = at_risk["chemical_historical_onset_rate"].fillna(
        bundle["fallback_historical_rate"]
    )
    X = at_risk[bundle["feature_columns"]]
    at_risk["phase1_production_score"] = bundle["model"].predict_proba(X)[:, 1]
    at_risk["phase3_shadow_score"] = np.nan  # stub -- activate once a Phase 3 shadow model is trained

    log_entry = at_risk[["chemical", "phase1_production_score", "phase3_shadow_score"]].copy()
    log_entry["month"] = yyyymm
    log_entry["scored_at"] = datetime.now().isoformat()

    if os.path.exists(LOG_PATH):
        existing = pd.read_csv(LOG_PATH)
        combined = pd.concat([existing, log_entry], ignore_index=True)
    else:
        combined = log_entry
    combined.to_csv(LOG_PATH, index=False)
    print(f"  Logged {len(log_entry)} predictions for {yyyymm} to {LOG_PATH}")


if __name__ == "__main__":
    config = load_config()

    if not config.get("url") or not config.get("next_month"):
        print(f"[{datetime.now()}] No new month configured -- nothing to do. "
              f"(This is the expected state most days.)")
        sys.exit(0)

    yyyymm = config["next_month"]
    url = config["url"]

    if already_processed(yyyymm):
        print(f"[{datetime.now()}] {yyyymm} already processed -- clearing config, nothing to do.")
        clear_config()
        sys.exit(0)

    print(f"[{datetime.now()}] New month configured: {yyyymm}. Running full pipeline...")
    run_pipeline_for_month(yyyymm, url)
    score_and_log(yyyymm)
    clear_config()
    print(f"[{datetime.now()}] Done with {yyyymm}. Config cleared -- waiting for next month's URL.")