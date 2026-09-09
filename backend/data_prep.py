"""
Data loading, cleaning, and feature engineering for the shipment delay model.
"""
import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["planned_shipment_date"])
    df = df.sort_values("planned_shipment_date").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features and interaction terms. No missing values were
    found in this dataset (see notebooks/eda.py), so no imputation is needed."""
    df = df.copy()

    # Calendar features — weak signal expected here since the dates in this
    # dataset are synthetically spread evenly across 2021-2045 (see README),
    # but included because a real-world version of this pipeline would need them.
    df["month"] = df["planned_shipment_date"].dt.month
    df["day_of_week"] = df["planned_shipment_date"].dt.dayofweek
    df["quarter"] = df["planned_shipment_date"].dt.quarter

    # Interaction feature suggested by the correlation analysis.
    # Note: an earlier version also added `num_blockers + num_dependencies`
    # as a "friction" feature, but that's an exact linear combination of two
    # columns already present here, which makes any linear model's design
    # matrix singular (rank-deficient). Removed for that reason — a boosted
    # tree model wouldn't care, but a linear model (our best performer on
    # this dataset) does.
    df["complexity_x_dependencies"] = df["feature_complexity"] * df["num_dependencies"]
    df["complexity_per_team_member"] = df["feature_complexity"] / df["team_size"]

    return df


FEATURE_COLUMNS = [
    "team_size",
    "feature_complexity",
    "num_dependencies",
    "sprint_length_weeks",
    "num_blockers",
    "holidays_in_sprint",
    "priority_encoded",
    "past_avg_delay_days",
    "estimated_bug_count",
    "month",
    "day_of_week",
    "quarter",
    "complexity_x_dependencies",
    "complexity_per_team_member",
]
TARGET_COLUMN = "delay_days"


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Hold out the most recent rows as test set, since this mimics real
    forecasting (train on the past, predict the future) rather than a random
    split, which would leak future patterns into training."""
    n_test = int(len(df) * test_frac)
    train_df = df.iloc[: len(df) - n_test]
    test_df = df.iloc[len(df) - n_test :]
    return train_df, test_df


def get_X_y(df: pd.DataFrame):
    return df[FEATURE_COLUMNS], df[TARGET_COLUMN]
