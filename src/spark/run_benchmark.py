"""
Phase 3, Task 3: runs the identical chemical-month aggregation (volume,
distinct practices, HHI) in Spark, DuckDB, and Polars against the SAME
Parquet data, recording real wall-clock time and peak memory for each.

WHY EACH ENGINE RUNS AS A SEPARATE SUBPROCESS, NOT IN-PROCESS:
Spark's actual computation happens in a JVM launched as a CHILD PROCESS of
the Python driver (via py4j) -- not inside the Python process itself. If
we measured only the Python process's own memory, we'd almost entirely
miss Spark's real memory use and the comparison would be meaningless.
Running every engine as its own subprocess and summing memory across that
subprocess's full process tree (parent + any children) makes the
measurement fair across all three, regardless of each engine's internal
architecture.

EXPECTED RESULT, STATED BEFORE RUNNING (so a "Spark loses" result isn't
mistaken for a bug): at this data size (one month, already Parquet-
landed, well within single-node RAM), DuckDB and Polars are expected to
win on both speed and memory -- Spark's JVM startup and distributed-
execution machinery is real overhead that only pays off once data
genuinely exceeds what a single process can hold. That crossover is what
scaling to more months is meant to demonstrate, not this one-month test.
"""
import argparse
import subprocess
import sys
import time
import psutil
import pandas as pd

POLL_INTERVAL_SEC = 0.2

WORKERS = {
    "spark": "src/spark/bench_spark.py",
    "duckdb": "src/spark/bench_duckdb.py",
    "polars": "src/spark/bench_polars.py",
}


def run_and_measure(engine: str, script: str, month: str) -> dict:
    print(f"\n=== Running {engine} ===")
    proc = subprocess.Popen(
        [sys.executable, script, "--month", month],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    peak_memory_mb = 0.0
    ps_proc = psutil.Process(proc.pid)

    while proc.poll() is None:
        try:
            total_rss = ps_proc.memory_info().rss
            for child in ps_proc.children(recursive=True):
                try:
                    total_rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak_memory_mb = max(peak_memory_mb, total_rss / 1e6)
        except psutil.NoSuchProcess:
            break
        time.sleep(POLL_INTERVAL_SEC)

    stdout, _ = proc.communicate()
    print(stdout)

    elapsed_sec = None
    for line in stdout.splitlines():
        if line.startswith("BENCH_RESULT"):
            for part in line.split():
                if part.startswith("elapsed_sec="):
                    elapsed_sec = float(part.split("=")[1])

    if elapsed_sec is None:
        print(f"  WARNING: {engine} did not report a clean result -- check its output above for an error.")

    return {"engine": engine, "elapsed_sec": elapsed_sec, "peak_memory_mb": round(peak_memory_mb, 1)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True)
    parser.add_argument("--out", default="data/interim/engine_benchmark.csv")
    args = parser.parse_args()

    results = []
    for engine, script in WORKERS.items():
        results.append(run_and_measure(engine, script, args.month))

    df = pd.DataFrame(results)
    print("\n=== Benchmark summary ===")
    print(df.to_string(index=False))

    df.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")