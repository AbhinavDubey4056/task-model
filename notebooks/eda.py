"""
Exploratory Data Analysis — Software Feature Shipment Dataset
Run: python notebooks/eda.py
Outputs: prints summary stats to console, saves plots to notebooks/eda_outputs/
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Software_feature_shipment_data_set.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "eda_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["planned_shipment_date"])

    print("=" * 60)
    print("SHAPE:", df.shape)
    print("=" * 60)
    print("\nMISSING VALUES:\n", df.isna().sum())
    print("\nDTYPES:\n", df.dtypes)
    print("\nDESCRIBE:\n", df.describe())

    # Outlier check via IQR
    print("\nOUTLIER COUNTS (1.5*IQR rule):")
    for col in df.select_dtypes(include="number").columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        print(f"  {col}: {n_out} outliers")

    # Correlation with target
    corr = df.corr(numeric_only=True)["delay_days"].sort_values(ascending=False)
    print("\nCORRELATION WITH delay_days:\n", corr)

    # Date range sanity check (dates in this dataset are synthetic/uniformly spread —
    # documented here rather than assumed, see README "Data notes")
    print("\nDATE RANGE:", df["planned_shipment_date"].min(), "to", df["planned_shipment_date"].max())
    print("ROWS PER YEAR:\n", df["planned_shipment_date"].dt.year.value_counts().sort_index())

    # Plots
    fig, ax = plt.subplots(figsize=(8, 6))
    corr.drop("delay_days").plot(kind="barh", ax=ax)
    ax.set_title("Feature correlation with delay_days")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "correlation_with_target.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    df["delay_days"].hist(bins=25, ax=ax)
    ax.set_title("Distribution of delay_days")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "delay_days_distribution.png"))

    print(f"\nPlots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
