"""
Minimal API server that exposes the trained model to the frontend.
Run: python3 backend/app.py
Then open http://localhost:5000 in a browser (it serves the frontend too).
"""
import os
import sys


BASE_DIR = os.path.dirname(__file__)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flask import Flask, request, jsonify, send_from_directory
from predict import predict_next_shipment

FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health")
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    required = [
        "planned_date", "team_size", "feature_complexity", "num_dependencies",
        "sprint_length_weeks", "num_blockers", "holidays_in_sprint",
        "priority_encoded", "past_avg_delay_days", "estimated_bug_count",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        raw_features = {
            "team_size": float(data["team_size"]),
            "feature_complexity": float(data["feature_complexity"]),
            "num_dependencies": float(data["num_dependencies"]),
            "sprint_length_weeks": float(data["sprint_length_weeks"]),
            "num_blockers": float(data["num_blockers"]),
            "holidays_in_sprint": int(data["holidays_in_sprint"]),
            "priority_encoded": int(data["priority_encoded"]),
            "past_avg_delay_days": float(data["past_avg_delay_days"]),
            "estimated_bug_count": float(data["estimated_bug_count"]),
        }
        result = predict_next_shipment(data["planned_date"], raw_features)
    except FileNotFoundError:
        return jsonify({"error": "No trained model found. Run `python backend/train.py` first."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

