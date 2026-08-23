"""
Phase 3, scaled: processes N months of raw EPD end to end, one at a time,
DELETING each month's raw CSV (~7.7GB) immediately after its Parquet
output is written AND verified correct -- so peak disk usage stays
roughly constant (~one month's raw file + accumulating ~50MB/month
Parquet outputs) instead of growing linearly with the number of months
processed. This is a deliberate ETL design choice, not a workaround: real
pipelines don't hoard raw source data once it's been transformed into the
format actually needed downstream.

SAFETY: a raw file is only deleted after its Parquet write is read back
and its row count is confirmed to exactly match the source CSV's row
count. If they don't match, the raw file is KEPT and the script stops --
never delete a file we haven't confirmed we no longer need.

Reuses ONE Spark session across all months (avoids ~13-16s of JVM
startup overhead per month that a fresh subprocess-per-month approach
would pay six times over).

Usage:
    python run_full_pipeline.py --months 202503 202504 202505 202506 202507 202508
"""
import argparse
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import sys
sys.path.insert(0, "src/spark")
from download_epd_month import download_month

RAW_DIR = "data/raw/epd_bulk"
PARQUET_DIR = "data/interim/epd_parquet"
FEATURES_DIR = "data/interim/epd_prescribing_features"
SPARK_TMP_DIR = "data/spark_tmp"


def build_spark_session() -> SparkSession:
    os.makedirs(SPARK_TMP_DIR, exist_ok=True)
    return (
        SparkSession.builder
        .appName("epd_full_pipeline")
        .master("local[4]")
        .config("spark.driver.memory", "5g")
        .config("spark.local.dir", os.path.abspath(SPARK_TMP_DIR))
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def process_one_month(spark: SparkSession, yyyymm: str) -> bool:
    print(f"\n{'='*60}\nProcessing {yyyymm}\n{'='*60}")
    t0 = time.time()

    raw_path = download_month(yyyymm)

    print(f"Reading raw CSV...")
    raw_df = spark.read.option("header", "true").csv(raw_path)
    raw_count = raw_df.count()
    print(f"  Raw row count: {raw_count:,}")

    df = raw_df.select(
        "YEAR_MONTH", "BNF_CHEMICAL_SUBSTANCE_CODE", "BNF_CHEMICAL_SUBSTANCE",
        "PRACTICE_CODE", "ICB_CODE",
        F.col("ITEMS").cast("double").alias("ITEMS"),
    ).coalesce(8)

    parquet_path = os.path.join(PARQUET_DIR, f"month={yyyymm}")
    print(f"Writing Parquet to {parquet_path}...")
    df.write.mode("overwrite").parquet(parquet_path)

    check_df = spark.read.parquet(parquet_path)
    parquet_count = check_df.count()
    print(f"  Parquet row count (read back): {parquet_count:,}")

    if raw_count != parquet_count:
        print(f"  MISMATCH: raw={raw_count:,} vs parquet={parquet_count:,} -- "
              f"KEEPING raw file, NOT deleting, stopping here for investigation.")
        return False

    print(f"  Row counts match. Computing chemical-month features...")
    practice_level = (
        check_df.groupBy("BNF_CHEMICAL_SUBSTANCE_CODE", "BNF_CHEMICAL_SUBSTANCE", "YEAR_MONTH", "PRACTICE_CODE")
        .agg(F.sum("ITEMS").alias("practice_items"))
    )
    chemical_level = (
        practice_level.groupBy("BNF_CHEMICAL_SUBSTANCE_CODE", "BNF_CHEMICAL_SUBSTANCE", "YEAR_MONTH")
        .agg(
            F.sum("practice_items").alias("total_items"),
            F.count("PRACTICE_CODE").alias("n_distinct_practices"),
            F.sum(F.pow(F.col("practice_items"), 2)).alias("sum_sq"),
        )
        .withColumn("hhi", F.col("sum_sq") / (F.col("total_items") * F.col("total_items")))
        .drop("sum_sq")
    )

    os.makedirs(FEATURES_DIR, exist_ok=True)
    features_csv = os.path.join(FEATURES_DIR, f"features_{yyyymm}.csv")
    chemical_level.toPandas().to_csv(features_csv, index=False)
    print(f"  Saved features to {features_csv}")

    raw_size_gb = os.path.getsize(raw_path) / 1e9
    os.remove(raw_path)
    print(f"  Deleted raw CSV ({raw_size_gb:.2f} GB freed) -- verified before delete, not assumed.")

    elapsed = time.time() - t0
    print(f"Done with {yyyymm} in {elapsed:.1f}s")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="+", required=True)
    args = parser.parse_args()

    spark = build_spark_session()
    results = {}
    try:
        for m in args.months:
            results[m] = process_one_month(spark, m)
            if not results[m]:
                print(f"\nStopping early due to mismatch on {m}. Fix before continuing.")
                break
    finally:
        spark.stop()

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for m, ok in results.items():
        print(f"  {m}: {'OK' if ok else 'FAILED -- see above'}")