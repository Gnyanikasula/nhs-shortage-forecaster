"""
Phase 3, Task 1: reads the raw monthly EPD CSV (~17M rows) and lands it as
partitioned Parquet -- aggregating nothing yet, just proving Spark can
process a file this size on a 16GB laptop without freezing it.

SAFETY CONFIG:
  - driver memory capped at 5GB, not Spark's tiny default.
  - spark.local.dir pointed at the D drive explicitly, so shuffle spill
    never fills a small system (C) drive.
  - local[4], not local[*] -- leaves CPU cores free for Windows/other apps.

Usage:
    python build_epd_parquet.py --month 202503
"""
import argparse
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

RAW_DIR = "data/raw/epd_bulk"
PARQUET_DIR = "data/interim/epd_parquet"
SPARK_TMP_DIR = "data/spark_tmp"


def build_spark_session() -> SparkSession:
    os.makedirs(SPARK_TMP_DIR, exist_ok=True)
    return (
        SparkSession.builder
        .appName("epd_parquet_landing")
        .master("local[4]")
        .config("spark.driver.memory", "5g")
        .config("spark.local.dir", os.path.abspath(SPARK_TMP_DIR))
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def land_month_as_parquet(spark: SparkSession, yyyymm: str):
    raw_path = os.path.join(RAW_DIR, f"epd_snomed_{yyyymm}.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"{raw_path} not found -- run download_epd_month.py --month {yyyymm} first."
        )

    print(f"Reading {raw_path} ...")
    t0 = time.time()

    df = (
        spark.read
        .option("header", "true")
        .csv(raw_path)
    )

    print(f"Row count (this forces a real read, not lazy): {df.count():,}")
    print(f"Columns: {df.columns}")

    df = df.select(
        "YEAR_MONTH",
        "BNF_CHEMICAL_SUBSTANCE_CODE",
        "BNF_CHEMICAL_SUBSTANCE",
        "PRACTICE_CODE",
        "ICB_CODE",
        F.col("ITEMS").cast("double").alias("ITEMS"),
    )

    # out_path = os.path.join(PARQUET_DIR, f"month={yyyymm}")
    # print(f"Writing partitioned Parquet to {out_path} ...")
    # df.write.mode("overwrite").parquet(out_path)
    out_path = os.path.join(PARQUET_DIR, f"month={yyyymm}")
    print(f"Writing partitioned Parquet to {out_path} ...")
    # coalesce(8) BEFORE writing -- shuffle.partitions only governs post-shuffle
    # partition count, and this job has no shuffle (no groupBy/join/sort), so
    # without this the output file count instead follows Spark's automatic
    # input-split of the source CSV (~118 tiny files for one month in testing --
    # a classic "small files" inefficiency that gets worse, not better, as more
    # months are added). coalesce merges partitions without a full shuffle,
    # which is the cheap way to control output file count here.
    df = df.coalesce(8)
    df.write.mode("overwrite").parquet(out_path)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    check = spark.read.parquet(out_path)
    print(f"Parquet row count (read back, confirms write succeeded): {check.count():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True, help="YYYYMM, e.g. 202503")
    args = parser.parse_args()

    spark = build_spark_session()
    try:
        land_month_as_parquet(spark, args.month)
    finally:
        spark.stop()