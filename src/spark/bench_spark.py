"""Benchmark worker: Spark. Run standalone (as a subprocess) so the
orchestrator can measure this process tree's memory in isolation."""
import argparse
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True)
    args = parser.parse_args()

    t0 = time.time()
    spark = (
        SparkSession.builder
        .appName("bench_spark")
        .master("local[4]")
        .config("spark.driver.memory", "5g")
        .config("spark.local.dir", os.path.abspath("data/spark_tmp"))
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    in_path = f"data/interim/epd_parquet/month={args.month}"
    df = spark.read.parquet(in_path)

    practice_level = (
        df.groupBy("BNF_CHEMICAL_SUBSTANCE_CODE", "BNF_CHEMICAL_SUBSTANCE", "YEAR_MONTH", "PRACTICE_CODE")
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
    )
    n = chemical_level.count()  # force execution -- Spark is lazy otherwise
    spark.stop()

    elapsed = time.time() - t0
    print(f"BENCH_RESULT engine=spark month={args.month} rows={n} elapsed_sec={elapsed:.3f}")