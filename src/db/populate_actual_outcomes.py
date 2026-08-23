"""
Phase 4: populates actual_outcomes in Postgres with real, final concession
history -- the ground truth needed to eventually evaluate the Phase 1
production model and Phase 3 shadow model against what actually happened.

DELIBERATELY USES THE ARCHIVE SPREADSHEET, NOT THE LIVE SCRAPER, AS THE
SOURCE OF TRUTH. The live CPE page publishes concessions incrementally
through the month ("2nd Update", "3rd Update"...) -- a chemical absent
from a mid-month scrape might still be added later. Recording that as a
False outcome would be genuine label noise, not just staleness. The
archive spreadsheet is CPE's own finalized, monthly-refreshed record --
already the trusted source for the entire Phase 1 panel -- so it's used
here too, rather than introducing two different sources of truth for the
same fact.

The live scraper (scrape_ncso_concessions.py) still has a real job: it
tells the SCORING step what's already been announced for the in-progress
current month (a feature), which is a different question from "what is
the FINAL truth for a completed month" (a label). Conflating those two
would corrupt the ground truth this script exists to protect. Wiring the
live scraper into scoring-time freshness is flagged as a real follow-up,
not solved here.

Safe to re-run: uses an upsert (ON CONFLICT DO UPDATE), so running this
again after the archive refreshes simply brings the database in sync
with whatever CPE has published since, rather than erroring on rows that
already exist.
"""
import os
import sys

from sqlalchemy.dialects.postgresql import insert as pg_insert

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from schema import get_engine, init_schema, actual_outcomes
from rebuild_panel_from_source import rebuild_everything


def populate_actual_outcomes(engine) -> int:
    print("Rebuilding panel from the archive (source of truth for outcomes)...")
    rebuilt = rebuild_everything()
    features = rebuilt["features"]

    rows = features[["chemical", "month", "on_concession"]].copy()
    rows["month"] = rows["month"].dt.strftime("%Y%m")
    records = rows.to_dict(orient="records")

    print(f"Upserting {len(records)} (chemical, month) outcome rows...")
    stmt = pg_insert(actual_outcomes).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["chemical", "month"],
        set_={"on_concession": stmt.excluded.on_concession},
    )
    with engine.begin() as conn:
        conn.execute(stmt)

    return len(records)


if __name__ == "__main__":
    engine = init_schema()
    n = populate_actual_outcomes(engine)
    print(f"Done. {n} outcome rows upserted into actual_outcomes.")