"""
Automated tests for the shipment delay pipeline.
Run with either:
    python3 -m unittest discover -s tests -v
    pytest tests/           (if pytest is installed — it can run unittest-style tests too)
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_prep import load_data, engineer_features, time_based_split, get_X_y, FEATURE_COLUMNS

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Software_feature_shipment_data_set.csv")


class TestDataPrep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_data(DATA_PATH)
        cls.df = engineer_features(cls.raw)

    def test_no_missing_values(self):
        self.assertEqual(self.raw.isna().sum().sum(), 0)

    def test_all_feature_columns_present(self):
        for col in FEATURE_COLUMNS:
            self.assertIn(col, self.df.columns)

    def test_feature_matrix_is_full_rank(self):
        # Regression test for the rank-deficiency bug found during
        # development (an engineered feature that exactly duplicated
        # information already in the dataset). This test exists so that
        # a future feature-engineering change can't silently reintroduce it.
        X, _ = get_X_y(self.df)
        rank = np.linalg.matrix_rank(X.values)
        self.assertEqual(rank, X.shape[1], "Feature matrix is rank-deficient — check for redundant engineered features")

    def test_time_based_split_is_chronological_and_non_overlapping(self):
        train_df, test_df = time_based_split(self.df, test_frac=0.2)
        self.assertEqual(len(train_df) + len(test_df), len(self.df))
        self.assertLessEqual(
            train_df["planned_shipment_date"].max(),
            test_df["planned_shipment_date"].min(),
            "Test set should come after the training set in time",
        )

    def test_target_is_non_negative(self):
        self.assertTrue((self.df["delay_days"] >= 0).all())


class TestPredict(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from predict import predict_next_shipment
        self.predict_next_shipment = predict_next_shipment

    def test_predict_returns_expected_keys(self):
        result = self.predict_next_shipment("2027-01-01", {
            "team_size": 10, "feature_complexity": 5.0, "num_dependencies": 2,
            "sprint_length_weeks": 2, "num_blockers": 0, "holidays_in_sprint": 0,
            "priority_encoded": 1, "past_avg_delay_days": 0.0, "estimated_bug_count": 4,
        })
        for key in ("planned_date", "predicted_delay_days", "predicted_shipment_date"):
            self.assertIn(key, result)

    def test_higher_complexity_predicts_more_delay(self):
        # Directional sanity check, not an exact-value check: given the
        # 0.82 correlation between feature_complexity and delay_days,
        # a much more complex feature (all else equal) should predict a
        # longer delay. This would catch e.g. a sign error in features
        # or an inverted train/test split.
        base_kwargs = dict(
            team_size=10, num_dependencies=2, sprint_length_weeks=2,
            num_blockers=0, holidays_in_sprint=0, priority_encoded=1,
            past_avg_delay_days=0.0, estimated_bug_count=4,
        )
        low = self.predict_next_shipment("2027-01-01", {**base_kwargs, "feature_complexity": 1.0})
        high = self.predict_next_shipment("2027-01-01", {**base_kwargs, "feature_complexity": 9.5})
        self.assertGreater(high["predicted_delay_days"], low["predicted_delay_days"])


if __name__ == "__main__":
    unittest.main()
