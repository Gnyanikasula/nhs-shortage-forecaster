from sqlalchemy import create_engine, text
import os

engine = create_engine(os.environ["DATABASE_URL"])
with engine.begin() as conn:
    conn.execute(text("ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS explanation VARCHAR"))
    conn.execute(text("ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS explanation_method VARCHAR"))
print("Columns added (or already existed).")