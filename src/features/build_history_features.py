"""
Builds the history-only features the plan's Phase 1 models need, from the
chemical-month panel built in build_chemical_panel.py.

LEAKAGE DISCIPLINE (read this before trusting any of it):
Every feature here is computed "as of month M" using only data from month M
and earlier. The label we're eventually predicting is on_concession(M+1)
given on_concession(M) == False. In particular:

  - chemical_historical_onset_rate(M) uses only onset transitions that
    happened strictly BEFORE month M -- not M itself. If it included M's
    own transition, the feature would partially encode the label we're
    trying to predict for that exact row. This is the one most likely to
    be gotten wrong by accident, so it's implemented with an explicit
    shift(1) before the cumulative sum, not just an expanding mean.

  - streak_length, time_since_last_concession, n_previous_episodes all use
    data up to and INCLUDING month M -- that's fine, not leakage, because
    the state at month M is genuinely observed/known at the point we're
    making a prediction about month M+1. Only "predicting one's own future
    transition using one's own future transition" is leakage; using the
    present to predict the future is the entire point.
"""

import pandas as pd
import numpy as np


def build_history_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["chemical", "month"]).reset_index(drop=True)

    prev_state = df.groupby("chemical")["on_concession"].shift(1)

    # ---- streak_length: consecutive months in the current state ----
    state_changed = (df["on_concession"] != prev_state) | prev_state.isna()
    streak_id = state_changed.groupby(df["chemical"]).cumsum()
    df["streak_length"] = df.groupby(["chemical", streak_id]).cumcount() + 1

    # ---- time_since_last_concession ----
    df["_month_idx"] = df["month"].dt.year * 12 + df["month"].dt.month
    df["_on_month_idx"] = np.where(df["on_concession"], df["_month_idx"], np.nan)
    df["_last_on_month_idx"] = df.groupby("chemical")["_on_month_idx"].ffill()
    df["time_since_last_concession"] = df["_month_idx"] - df["_last_on_month_idx"]
    df["never_seen_on_concession"] = df["_last_on_month_idx"].isna()

    # ---- n_previous_episodes ----
    # prev_state comes out of groupby().shift() with dtype 'object' (NaN mixed
    # into a bool column upcasts it), and Python's `~` on an object-dtype bool
    # does BITWISE not (~True == -2, ~False == -1 -- both truthy!), not logical
    # not. Must force back to real bool dtype before negating, or this silently
    # returns "always True" and episode_start collapses to just on_concession.
    prev_state_bool = prev_state.fillna(False).astype(bool)
    episode_start = df["on_concession"] & (~prev_state_bool)
    df["n_previous_episodes"] = episode_start.groupby(df["chemical"]).cumsum()

    # ---- chemical_historical_onset_rate: expanding rate, STRICTLY PRIOR months only ----
    next_state = df.groupby("chemical")["on_concession"].shift(-1)
    is_valid_transition = (~df["on_concession"]) & next_state.notna()
    onset_event = np.where(is_valid_transition, next_state.astype(float), np.nan)

    df["_clear_transition_ind"] = is_valid_transition.astype(float)
    df["_onset_ind"] = pd.Series(onset_event, index=df.index).fillna(0) * df["_clear_transition_ind"]

    clear_shifted = df.groupby("chemical")["_clear_transition_ind"].shift(1)
    onset_shifted = df.groupby("chemical")["_onset_ind"].shift(1)
    cum_clear = clear_shifted.groupby(df["chemical"]).cumsum()
    cum_onset = onset_shifted.groupby(df["chemical"]).cumsum()

    df["chemical_historical_onset_rate"] = cum_onset / cum_clear
    df["n_prior_valid_transitions"] = cum_clear

    # ---- month of year (seasonality) ----
    df["month_of_year"] = df["month"].dt.month

    # the actual label for this row
    df["label_onset_next_month"] = np.where(is_valid_transition, next_state.astype(float), np.nan)

    df = df.drop(columns=["_month_idx", "_on_month_idx", "_last_on_month_idx",
                           "_clear_transition_ind", "_onset_ind"])
    return df


if __name__ == "__main__":
    panel = pd.read_parquet("data/interim/chemical_month_panel.parquet")
    features = build_history_features(panel)

    print(f"Built features for {len(features):,} rows, {features['chemical'].nunique()} chemicals")
    print()

    sample = features[features["chemical"] == "Atomoxetine"].tail(12)
    print("Atomoxetine, last 12 months -- eyeball this against the raw archive:")
    print(sample[["month", "on_concession", "streak_length",
                   "time_since_last_concession", "n_previous_episodes",
                   "chemical_historical_onset_rate", "n_prior_valid_transitions"]].to_string(index=False))

    features.to_parquet("data/interim/chemical_features.parquet")
    print(f"\nSaved to data/interim/chemical_features.parquet")