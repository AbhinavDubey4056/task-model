"""
Predict the delay (and resulting next shipment date) for a new feature.
Run: python src/predict.py
"""
import os
import joblib
import pandas as pd

from data_prep import engineer_features, FEATURE_COLUMNS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "shipment_delay_model.joblib")


def predict_next_shipment(planned_date: str, raw_features: dict) -> dict:
    """
    planned_date: e.g. "2026-11-02"
    raw_features: dict with team_size, feature_complexity, num_dependencies,
                  sprint_length_weeks, num_blockers, holidays_in_sprint,
                  priority_encoded, past_avg_delay_days, estimated_bug_count
    """
    model = joblib.load(MODEL_PATH)

    row = {"planned_shipment_date": pd.to_datetime(planned_date), **raw_features}
    df = pd.DataFrame([row])
    df = engineer_features(df)

    predicted_delay = model.predict(df[FEATURE_COLUMNS])[0]
    predicted_date = df["planned_shipment_date"].iloc[0] + pd.Timedelta(days=predicted_delay)

    return {
        "planned_date": planned_date,
        "predicted_delay_days": round(float(predicted_delay), 1),
        "predicted_shipment_date": predicted_date.strftime("%Y-%m-%d"),
    }


if __name__ == "__main__":
    example = {
        "team_size": 12,
        "feature_complexity": 6.5,
        "num_dependencies": 3,
        "sprint_length_weeks": 3,
        "num_blockers": 1,
        "holidays_in_sprint": 0,
        "priority_encoded": 1,
        "past_avg_delay_days": 0.5,
        "estimated_bug_count": 5,
    }
    result = predict_next_shipment("2026-11-02", example)
    print(result)
