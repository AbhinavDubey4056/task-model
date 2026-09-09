
import os
import json
import warnings
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import load_data, engineer_features, time_based_split, get_X_y

try:
    from xgboost import XGBRegressor
    MODEL_BACKEND = "xgboost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor as XGBRegressor
    MODEL_BACKEND = "sklearn_hgbr (xgboost not installed — run `pip install xgboost` for the intended model)"

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Software_feature_shipment_data_set.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul.*",
    category=RuntimeWarning,
)


def evaluate(name, y_true, y_pred, results):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    results[name] = {"MAE": round(mae, 3), "RMSE": round(rmse, 3)}
    print(f"{name:35s} MAE={mae:.3f}  RMSE={rmse:.3f}")


def main():
    print(f"Model backend: {MODEL_BACKEND}\n")

    df = engineer_features(load_data(DATA_PATH))
    train_df, test_df = time_based_split(df, test_frac=0.2)
    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)

    results = {}


    dummy = DummyRegressor(strategy="median").fit(X_train, y_train)
    evaluate("Baseline (median)", y_test, dummy.predict(X_test), results)

    lin = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_train, y_train)
    evaluate("Baseline (ridge regression)", y_test, lin.predict(X_test), results)


    if MODEL_BACKEND == "xgboost":
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5],
        }
        base_model = XGBRegressor(objective="reg:absoluteerror", random_state=42)
    else:
        param_dist = {
            "max_iter": [100, 200, 300],
            "max_depth": [3, 4, 5, 6, None],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "min_samples_leaf": [10, 20, 30],
        }
        base_model = XGBRegressor(loss="absolute_error", random_state=42)

    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=25,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=42,

        n_jobs=1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("\nBest params:", search.best_params_)

    evaluate("Tuned gradient boosting model", y_test, best_model.predict(X_test), results)

    feature_names = X_train.columns.tolist()
    try:
        import shap
        explainer = shap.Explainer(best_model)
        shap_values = explainer(X_test)
        importance = np.abs(shap_values.values).mean(axis=0)
    except ImportError:
        importance = getattr(best_model, "feature_importances_", None)

    if importance is not None:
        ranked = sorted(zip(feature_names, importance), key=lambda x: -x[1])
        print("\nFeature importance:")
        for name, val in ranked:
            print(f"  {name:30s} {val:.4f}")


    joblib.dump(best_model, os.path.join(MODEL_DIR, "shipment_delay_model.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nModel and metrics saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
