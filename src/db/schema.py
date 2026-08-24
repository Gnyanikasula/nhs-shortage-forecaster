"""
Database schema for the cloud-ready pipeline. Written against standard
SQLAlchemy Core so the SAME code works unchanged whether DATABASE_URL
points at a local Docker Postgres container (free, no signup, for
testing today) or a real Neon Postgres instance (free, no card, for
when this actually goes live) -- only the connection string differs.

Deliberately does NOT store the 46GB of raw EPD data or the 79-month
concession archive -- those are cheap to re-fetch/rebuild from source
each run. Only the state an ephemeral container can't cheaply
reconstruct lives here.
"""
import os
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, String, Float, DateTime, Boolean, UniqueConstraint,
)
from sqlalchemy.sql import func

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///data/interim/local_dev.db")

metadata = MetaData()

epd_prescribing_features = Table(
    "epd_prescribing_features", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("bnf_chemical_substance_code", String, nullable=False),
    Column("bnf_chemical_substance", String, nullable=False),
    Column("year_month", String, nullable=False),
    Column("total_items", Float, nullable=False),
    Column("n_distinct_practices", Integer, nullable=False),
    Column("hhi", Float, nullable=False),
    Column("computed_at", DateTime, server_default=func.now()),
    UniqueConstraint("bnf_chemical_substance_code", "year_month", name="uq_chem_month"),
)

# prediction_log = Table(
#     "prediction_log", metadata,
#     Column("id", Integer, primary_key=True, autoincrement=True),
#     Column("chemical", String, nullable=False),
#     Column("month", String, nullable=False),
#     Column("phase1_production_score", Float, nullable=True),
#     Column("phase3_shadow_score", Float, nullable=True),
#     Column("scored_at", DateTime, server_default=func.now()),
# )
# prediction_log = Table(
#     "prediction_log", metadata,
#     Column("id", Integer, primary_key=True, autoincrement=True),
#     Column("chemical", String, nullable=False),
#     Column("month", String, nullable=False),
#     Column("phase1_production_score", Float, nullable=True),
#     Column("phase3_shadow_score", Float, nullable=True),
#     Column("scored_at", DateTime, server_default=func.now(), onupdate=func.now()),
#     UniqueConstraint("chemical", "month", name="uq_chem_prediction_month"),
# )
prediction_log = Table(
    "prediction_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("chemical", String, nullable=False),
    Column("month", String, nullable=False),
    Column("phase1_production_score", Float, nullable=True),
    Column("phase3_shadow_score", Float, nullable=True),
    Column("scored_at", DateTime, server_default=func.now(), onupdate=func.now()),
    Column("explanation", String, nullable=True),
    Column("explanation_method", String, nullable=True),
    UniqueConstraint("chemical", "month", name="uq_chem_prediction_month"),
)

actual_outcomes = Table(
    "actual_outcomes", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("chemical", String, nullable=False),
    Column("month", String, nullable=False),
    Column("on_concession", Boolean, nullable=False),
    Column("recorded_at", DateTime, server_default=func.now()),
    UniqueConstraint("chemical", "month", name="uq_chem_outcome_month"),
)

pipeline_state = Table(
    "pipeline_state", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("next_month", String, nullable=True),
    Column("url", String, nullable=True),
    Column("updated_at", DateTime, server_default=func.now(), onupdate=func.now()),
)


def get_engine():
    return create_engine(DATABASE_URL)


def init_schema(engine=None):
    engine = engine or get_engine()
    metadata.create_all(engine)
    return engine


# if __name__ == "__main__":
#     print(f"Connecting to: {DATABASE_URL}")
def _redact(url: str) -> str:
    """Never print a connection string with its password intact --
    this line runs unchanged once DATABASE_URL points at a real
    production database, and Docker/GitHub Actions logs are not private
    by default. Cheap to fix now, costly to forget later."""
    if "@" in url and "://" in url:
        scheme_and_creds, rest = url.split("@", 1)
        scheme, _ = scheme_and_creds.split("://", 1)
        return f"{scheme}://***:***@{rest}"
    return url


if __name__ == "__main__":
    print(f"Connecting to: {_redact(DATABASE_URL)}")
    engine = init_schema()
    print("Schema created successfully. Tables:")
    for t in metadata.tables:
        print(f"  {t}")