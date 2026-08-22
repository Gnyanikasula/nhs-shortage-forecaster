"""
Phase 3, Task 2: the actual feature computation the whole phase exists for --
per chemical, per month, three things impossible to get from the
pre-aggregated OpenPrescribing API:

  - total_items:            national volume (also computable from the API --
                             included here as a cross-check against known
                             values, not as a new feature)
  - n_distinct_practices:   how many distinct GP practices prescribed this
                             chemical that month -- row-level only, the API
                             has no per-practice breakdown at all
  - hhi:                    Herfindahl-Hirschman Index of practice-level
                             concentration. HHI = sum((practice_share)^2)
                             for practice_share = practice_items/total_items.
                             Ranges 0 (spread evenly across many practices)
                             to 1 (all prescriptions from one practice).

TWO-STAGE GROUPBY, NOT A WINDOW FUNCTION OR SELF-JOIN:
HHI needs each practice's share of the chemical's total that month. The
naive approach joins the chemical-level total back onto every row to
compute a per-row share, then squares and sums -- an expensive join on
18M rows. Avoided here using the identity:
    sum((x_i / T)^2) = sum(x_i^2) / T^2
So stage 1 computes each practice's item sum (small, ~7000 practices x
~1500 chemicals), stage 2 sums both the values AND their squares in one
groupBy, then divides once at the end. No join needed.

VALIDATION: cross-checked total_items against epd_adhd_full.csv's already-
verified OpenPrescribing values for March 2025 (e.g. Atomoxetine
hydrochloride = 15,242) -- if this job's total_items matches, the whole
pipeline (download -> Parquet -> aggregation) is confirmed correct end to
end, not just "ran without error".
"""
import argparse
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PARQUET_DIR = "data/interim/epd_parquet"
OUTPUT_DIR = "data/interim/epd_prescribing_features"
SPARK_TMP_DIR = "data/spark_tmp"


def build_spark_session() -> SparkSession:
    os.makedirs(SPARK_TMP_DIR, exist_ok=True)
    return (
        SparkSession.builder
        .appName("epd_prescribing_features")
        .master("local[4]")
        .config("spark.driver.memory", "5g")
        .config("spark.local.dir", os.path.abspath(SPARK_TMP_DIR))
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def compute_features(spark: SparkSession, yyyymm: str):
    in_path = os.path.join(PARQUET_DIR, f"month={yyyymm}")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"{in_path} not found -- run build_epd_parquet.py --month {yyyymm} first.")

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
            F.sum(F.pow(F.col("practice_items"), 2)).alias("sum_sq_practice_items"),
        )
        .withColumn(
            "hhi",
            F.when(
                F.col("total_items") > 0,
                F.col("sum_sq_practice_items") / (F.col("total_items") * F.col("total_items")),
            ).otherwise(None),
        )
        .drop("sum_sq_practice_items")
    )

    return chemical_level


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True, help="YYYYMM, e.g. 202503")
    args = parser.parse_args()

    spark = build_spark_session()
    try:
        result = compute_features(spark, args.month)

        print(f"Computed features for {result.count()} chemicals in {args.month}")
        print()

        known_march_2025 = {
            "0404000L0": ("Dexamfetamine sulfate", 17383),
            "0404000M0": ("Methylphenidate hydrochloride", 160783),
            "0404000S0": ("Atomoxetine hydrochloride", 15242),
            "0404000U0": ("Lisdexamfetamine dimesylate", 74947),
            "0404000V0": ("Guanfacine", 7783),
        }
        if args.month == "202503":
            print("=== Cross-check against known OpenPrescribing values (March 2025) ===")
            pdf = result.filter(F.col("BNF_CHEMICAL_SUBSTANCE_CODE").isin(list(known_march_2025.keys()))).toPandas()
            for _, row in pdf.iterrows():
                expected_name, expected_items = known_march_2025[row["BNF_CHEMICAL_SUBSTANCE_CODE"]]
                match = "MATCH" if int(row["total_items"]) == expected_items else "MISMATCH -- investigate"
                print(f"  {row['BNF_CHEMICAL_SUBSTANCE']}: computed={int(row['total_items'])}  "
                      f"expected={expected_items}  [{match}]")
                print(f"    n_distinct_practices={int(row['n_distinct_practices'])}  hhi={row['hhi']:.5f}")
            print()

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_csv = os.path.join(OUTPUT_DIR, f"features_{args.month}.csv")
        result.toPandas().to_csv(out_csv, index=False)
        print(f"Saved to {out_csv}")

    finally:
        spark.stop()