"""Regression tests for data repairs, split boundaries, and final artifacts."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.spotter_freight.modeling import FreightCleaner, regression_metrics
from src.spotter_freight.train import split_by_date


ROOT = Path(__file__).resolve().parents[1]


class CleaningTests(unittest.TestCase):
    def test_weight_and_market_repairs_preserve_flags(self) -> None:
        training = pd.DataFrame(
            {
                "pickup": ["A", "A", "B"],
                "delivery": ["B", "C", "A"],
                "pickup_lat": [35.0, 35.0, 36.0],
                "pickup_lon": [-90.0, -90.0, -91.0],
                "delivery_lat": [36.0, 37.0, 35.0],
                "delivery_lon": [-91.0, -92.0, -90.0],
                "distance": [100.0, 200.0, 100.0],
                "equipment": ["Dry Van", "Dry Van", "Reefer"],
                "weight": [10_000.0, 20_000.0, 30_000.0],
                "date": ["2025-01-01", "2025-01-01", "2025-01-02"],
                "market_index": [0.9, 1.1, 1.0],
                "quote_signal": [2.0, 2.1, 1.9],
            }
        )
        test = training.iloc[:2].copy()
        test.loc[0, "weight"] = -12_000.0
        test.loc[1, "weight"] = np.nan
        test.loc[0, "market_index"] = np.nan

        cleaned = FreightCleaner().fit(training).transform(test)

        self.assertEqual(cleaned.loc[0, "weight"], 12_000.0)
        self.assertEqual(cleaned.loc[0, "weight_was_negative"], 1.0)
        self.assertEqual(cleaned.loc[1, "weight"], 15_000.0)
        self.assertEqual(cleaned.loc[1, "weight_was_missing"], 1.0)
        self.assertEqual(cleaned.loc[0, "market_index"], 1.1)
        self.assertEqual(cleaned.loc[0, "market_index_was_missing"], 1.0)

    def test_metric_definitions(self) -> None:
        metrics = regression_metrics(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
        self.assertAlmostEqual(metrics["mae"], 15.0)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(250.0))
        self.assertAlmostEqual(metrics["mape_percent"], 10.0)


class RepositoryArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = pd.read_csv(ROOT / "data" / "train_test.csv")
        cls.validation = pd.read_csv(ROOT / "data" / "validation.csv")
        cls.predictions = pd.read_csv(ROOT / "validation_predictions.csv")
        cls.december = pd.read_csv(ROOT / "data" / "december_chart_inputs.csv")
        cls.december_template = pd.read_csv(
            ROOT / "data" / "december_chart_inputs_template.csv"
        )

    def test_chronological_partitions(self) -> None:
        inner, tuning, holdout = split_by_date(self.train)
        self.assertEqual((len(inner), len(tuning), len(holdout)), (33_718, 4_759, 9_523))
        self.assertLess(pd.to_datetime(inner["date"]).max(), pd.to_datetime(tuning["date"]).min())
        self.assertLess(pd.to_datetime(tuning["date"]).max(), pd.to_datetime(holdout["date"]).min())
        self.assertEqual(pd.to_datetime(holdout["date"]).nunique(), 61)

    def test_submission_schema_ids_and_values(self) -> None:
        self.assertEqual(self.predictions.columns.tolist(), ["load_id", "predicted_rate"])
        self.assertEqual(len(self.predictions), 12_000)
        self.assertTrue(self.predictions["load_id"].equals(self.validation["load_id"]))
        self.assertFalse(self.predictions.isna().any().any())
        self.assertTrue(np.isfinite(self.predictions["predicted_rate"]).all())
        self.assertTrue((self.predictions["predicted_rate"] > 0).all())

    def test_december_schema_and_fixed_inputs(self) -> None:
        expected_columns = [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "predicted_rate",
        ]
        self.assertEqual(self.december.columns.tolist(), expected_columns)
        self.assertEqual(self.december_template.columns.tolist(), expected_columns)
        self.assertEqual(len(self.december), 31)
        self.assertEqual(len(self.december_template), 31)
        self.assertTrue(self.december_template["predicted_rate"].isna().all())
        self.assertTrue(self.december["pickup"].eq("Lexington").all())
        self.assertTrue(self.december["delivery"].eq("Fort Wayne").all())
        self.assertTrue(self.december["distance"].eq(360).all())
        self.assertTrue(self.december["equipment"].eq("Dry Van").all())
        self.assertTrue(self.december["weight"].eq(32_000).all())
        self.assertTrue((self.december["predicted_rate"] > 0).all())
        expected_dates = pd.date_range("2025-12-01", "2025-12-31")
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(pd.to_datetime(self.december["date"])),
            expected_dates,
            check_names=False,
        )


if __name__ == "__main__":
    unittest.main()
