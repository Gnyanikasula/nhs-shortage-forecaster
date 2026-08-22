"""Benchmark worker: Polars. Run standalone so the orchestrator can
measure this process's memory in isolation."""
import argparse
import time
import polars as pl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True)
    args = parser.parse_args()

    t0 = time.time()
    glob_path = f"data/interim/epd_parquet/month={args.month}/*.parquet"

    practice_level = (
        pl.scan_parquet(glob_path)
        .group_by(["BNF_CHEMICAL_SUBSTANCE_CODE", "BNF_CHEMICAL_SUBSTANCE", "YEAR_MONTH", "PRACTICE_CODE"])
        .agg(pl.col("ITEMS").sum().alias("practice_items"))
    )
    chemical_level = (
        practice_level.group_by(["BNF_CHEMICAL_SUBSTANCE_CODE", "BNF_CHEMICAL_SUBSTANCE", "YEAR_MONTH"])
        .agg([
            pl.col("practice_items").sum().alias("total_items"),
            pl.col("practice_items").count().alias("n_distinct_practices"),
            (pl.col("practice_items") ** 2).sum().alias("sum_sq"),
        ])
        .with_columns((pl.col("sum_sq") / (pl.col("total_items") ** 2)).alias("hhi"))
        .drop("sum_sq")
    )
    # result = chemical_level.collect()  # forces execution -- lazy otherwise
    result = chemical_level.collect(engine="streaming")  # forces execution -- lazy otherwise

    elapsed = time.time() - t0
    print(f"BENCH_RESULT engine=polars month={args.month} rows={len(result)} elapsed_sec={elapsed:.3f}")