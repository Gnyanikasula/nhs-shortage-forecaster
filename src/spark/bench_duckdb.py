"""Benchmark worker: DuckDB. Run standalone so the orchestrator can
measure this process's memory in isolation."""
import argparse
import time
import duckdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True)
    args = parser.parse_args()

    t0 = time.time()
    glob_path = f"data/interim/epd_parquet/month={args.month}/*.parquet"

    result = duckdb.sql(f"""
        WITH practice_level AS (
            SELECT BNF_CHEMICAL_SUBSTANCE_CODE, BNF_CHEMICAL_SUBSTANCE, YEAR_MONTH,
                   PRACTICE_CODE, SUM(ITEMS) AS practice_items
            FROM read_parquet('{glob_path}')
            GROUP BY 1, 2, 3, 4
        )
        SELECT BNF_CHEMICAL_SUBSTANCE_CODE, BNF_CHEMICAL_SUBSTANCE, YEAR_MONTH,
               SUM(practice_items) AS total_items,
               COUNT(PRACTICE_CODE) AS n_distinct_practices,
               SUM(POWER(practice_items, 2)) / POWER(SUM(practice_items), 2) AS hhi
        FROM practice_level
        GROUP BY 1, 2, 3
    """).fetchall()

    elapsed = time.time() - t0
    print(f"BENCH_RESULT engine=duckdb month={args.month} rows={len(result)} elapsed_sec={elapsed:.3f}")