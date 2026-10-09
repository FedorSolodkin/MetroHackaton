"""
Unit-тесты для построителя признаков (src/models/feature_builder.py).
Проверяют отсутствие утечек данных (Zero Leakage), корректность сдвигов и инференс-признаков.
"""

import unittest
import numpy as np
import pandas as pd

from src.models.baseline import MetroBaselineModel
from src.models.feature_builder import MetroFeatureBuilder, FEATURE_COLUMNS


class TestMetroFeatureBuilder(unittest.TestCase):

    def setUp(self):
        # Синтетический датасет для 3 станций (включая 111, 120, 129) за 3 дня
        records = []
        d0 = pd.Timestamp("2026-05-04 05:30:00") # Понедельник
        for day_offset in range(3):
            base_dt = d0 + pd.Timedelta(days=day_offset)
            for s in [111, 120, 129]:
                for slot in range(10, 86):
                    dt = base_dt + pd.Timedelta(minutes=(slot - 10) * 15)
                    records.append({
                        "datetime": dt,
                        "station_code": s,
                        "passengers": 150.0 + (s - 110) * 5 + slot
                    })
        self.df = pd.DataFrame(records)
        self.baseline = MetroBaselineModel()
        self.baseline.fit(self.df)
        self.builder = MetroFeatureBuilder(baseline_model=self.baseline)

    def test_training_dataset_construction(self):
        for H in [15, 30, 60, 120]:
            dataset_h, feat_cols = self.builder.build_training_dataset_for_horizon(self.df, horizon_min=H)
            self.assertGreater(len(dataset_h), 0)
            self.assertEqual(feat_cols, FEATURE_COLUMNS)
            for col in feat_cols:
                self.assertIn(col, dataset_h.columns)
            
            # Проверка, что целевое время ровно на H минут позже момента T
            k = H // 15
            expected_diff = pd.Timedelta(minutes=H)
            actual_diffs = (dataset_h["target_datetime"] - dataset_h["datetime"])
            self.assertTrue((actual_diffs == expected_diff).all(), f"Разница во времени для H={H} некорректна")

    def test_inference_features_shape(self):
        current_time = pd.Timestamp("2026-05-05 12:00:00")
        inf_df = self.builder.build_inference_features(current_time, self.df, horizon_min=30)
        # Должно быть 19 станций
        self.assertEqual(len(inf_df), 19)
        self.assertEqual(inf_df["target_datetime"].iloc[0], current_time + pd.Timedelta(minutes=30))
        for col in FEATURE_COLUMNS:
            self.assertIn(col, inf_df.columns)
            self.assertFalse(inf_df[col].isna().any(), f"Колонка {col} содержит NaN!")


if __name__ == "__main__":
    unittest.main()
