"""
Streamlit dashboard for the NHS Drug Shortage Early Warning System.
Reads directly from Postgres (same data the API serves) -- not calling
the API over HTTP, since both run in the same deployment and a direct
DB read is simpler than adding a network hop for no benefit.
"""
import os
import sys

import pandas as pd
import streamlit as st
from sqlalchemy import select, func

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "db"))

from schema import get_engine, prediction_log, actual_outcomes

st.set_page_config(page_title="NHS Shortage Early Warning", layout="wide")

engine = get_engine()


@st.cache_data(ttl=300)
def load_latest_predictions():
    with engine.connect() as conn:
        latest_month = conn.execute(select(func.max(prediction_log.c.month))).scalar()
        if latest_month is None:
            return None, pd.DataFrame()
        rows = conn.execute(
            select(prediction_log)
            .where(prediction_log.c.month == latest_month)
            .order_by(prediction_log.c.phase1_production_score.desc())
        ).fetchall()
    df = pd.DataFrame(rows, columns=prediction_log.columns.keys())
    return latest_month, df


@st.cache_data(ttl=300)
def load_stats():
    with engine.connect() as conn:
        n_outcomes = conn.execute(select(func.count()).select_from(actual_outcomes)).scalar()
        n_predictions = conn.execute(select(func.count()).select_from(prediction_log)).scalar()
    return n_outcomes, n_predictions


@st.cache_data(ttl=300)
def load_chemical_history(chemical: str):
    with engine.connect() as conn:
        rows = conn.execute(
            select(actual_outcomes)
            .where(actual_outcomes.c.chemical == chemical)
            .order_by(actual_outcomes.c.month)
        ).fetchall()
    return pd.DataFrame(rows, columns=actual_outcomes.columns.keys())


st.title("NHS Drug Shortage Early Warning System")
st.caption(
    "History-only XGBoost model, walk-forward validated (PR-AUC 0.096 vs 0.064 base rate). "
    "Predicts probability of a chemical entering NHS price concession next month."
)

n_outcomes, n_predictions = load_stats()
col1, col2, col3 = st.columns(3)
col1.metric("Ground-truth outcome rows", f"{n_outcomes:,}")
col2.metric("Logged predictions", f"{n_predictions:,}")

latest_month, df = load_latest_predictions()

if df.empty:
    st.warning("No predictions logged yet. Run src/db/score_latest_month.py first.")
else:
    col3.metric("Latest scored month", latest_month)

    st.subheader(f"Top risk chemicals -- {latest_month}")
    top_n = st.slider("Number of chemicals to show", min_value=5, max_value=50, value=20)

    display_df = df.head(top_n)[["chemical", "phase1_production_score"]].copy()
    display_df.columns = ["Chemical", "Onset risk (next month)"]
    display_df["Onset risk (next month)"] = display_df["Onset risk (next month)"].round(4)

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.bar_chart(display_df.set_index("Chemical")["Onset risk (next month)"])

    st.subheader("Chemical concession history")
    selected_chemical = st.selectbox("Select a chemical to view its real history", df["chemical"].tolist())
    history_df = load_chemical_history(selected_chemical)

    if not history_df.empty:
        history_df["month"] = pd.to_datetime(history_df["month"], format="%Y%m")
        history_df = history_df.sort_values("month")
        st.line_chart(history_df.set_index("month")["on_concession"].astype(int))
        st.caption("1 = on price concession that month, 0 = clear. Real historical data, not predicted.")
    else:
        st.info(f"No outcome history found for {selected_chemical}.")

st.divider()
st.caption(
    "Data sources: Community Pharmacy England (concession archive), NHSBSA (prescribing data). "
    "Model retrained on every pipeline run from source data -- no persisted model file. "
    "See project repository for full methodology and honestly-reported results across all phases."
)