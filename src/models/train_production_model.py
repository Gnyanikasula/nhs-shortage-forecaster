"""
Trains the Phase 1 (history-only) XGBoost model on ALL available history
and saves it -- the missing piece that turns "a model we evaluated inside
a walk-forward loop" into "a model that can score a brand new month".

This is deliberately separate from baselines_walk_forward.py, which
retrains fresh at every walk-forward step (correct FOR EVALUATION, since
each step must only see data before its test month). For actual
production scoring of a new month, we want ONE model trained on
everything known so far -- there's no leakage risk in using all history
to score a month that hasn't happened yet.

Retrain monthly: as each new month's true outcome becomes known, it
should be folded into the next training run, not left stale.
"""
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "streak_length", "time_since_last_concession_filled", "n_previous_episodes",
    "n_prior_valid_transitions", "month_sin", "month_cos",
    "chemical_historical_onset_rate",
]

MODEL_PATH = "data/interim/phase1_production_model.joblib"


def prepare_frame(features: pd.DataFrame) -> pd.DataFrame:
    df = features[features["label_onset_next_month"].notna()].copy()
    df["time_since_last_concession_filled"] = df["time_since_last_concession"].fillna(999)
    df["month_sin"] = np.sin(2 * np.pi * df["month_of_year"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_of_year"] / 12)
    overall_rate = df["label_onset_next_month"].mean()
    df["chemical_historical_onset_rate"] = df["chemical_historical_onset_rate"].fillna(overall_rate)
    return df, overall_rate


def train_production_model(features_path: str = "data/interim/chemical_features.parquet") -> dict:
    features = pd.read_parquet(features_path)
    df, overall_rate = prepare_frame(features)

    X = df[FEATURE_COLUMNS]
    y = df["label_onset_next_month"].astype(int)

    print(f"Training on ALL available history: {len(df)} rows, {y.sum()} onset events, "
          f"{df['month'].min()} to {df['month'].max()}")

    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                           eval_metric="aucpr", random_state=42)
    model.fit(X, y)

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "trained_through_month": str(df["month"].max()),
        "fallback_historical_rate": overall_rate,
        "n_training_rows": len(df),
        "n_training_events": int(y.sum()),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")
    return bundle


if __name__ == "__main__":
    train_production_model()