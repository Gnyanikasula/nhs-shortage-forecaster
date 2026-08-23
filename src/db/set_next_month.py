"""
The human's monthly task, now targeting the database instead of a local
JSON file: check NHSBSA's dataset page, find the new month's download
URL, run this once.

Usage:
    python set_next_month.py --month 202609 --url "https://.../epd_snomed_202609.csv"
"""
import argparse
import sys
from sqlalchemy import insert

sys.path.insert(0, "src/db")
from schema import get_engine, init_schema, pipeline_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True, help="YYYYMM, e.g. 202609")
    parser.add_argument("--url", required=True)
    args = parser.parse_args()

    engine = init_schema()
    with engine.begin() as conn:
        conn.execute(insert(pipeline_state).values(next_month=args.month, url=args.url))

    print(f"Set next_month={args.month} in the database. "
          f"The next scheduled run of run_monthly_update_db.py will pick it up.")