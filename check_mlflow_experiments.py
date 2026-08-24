from sqlalchemy import create_engine, text
import os

engine = create_engine(os.environ["DATABASE_URL"])
with engine.connect() as conn:
    rows = conn.execute(text("SELECT name FROM experiments")).fetchall()
    print(rows)