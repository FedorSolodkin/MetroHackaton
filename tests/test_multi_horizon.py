"""
Unit-тесты для мультигоризонтных моделей LightGBM (src/models/train_multi_horizon.py).
Проверяют обучение, монотонность квантилей P10 <= P50 <= P90, сохранение и загрузку.
"""

import unittest
import os
import tempfile
import numpy as np
import pandas as pd

from src.models.baseline import MetroBaselineModel
from src.models.feature_builder import MetroFeatureBuilder
from src.models.train_multi_horizon import MultiHorizonFlowModel


class TestMultiHorizonFlowModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Генерируем компактный синтетический датасет для тестирования обучения
        records = []
        d0 = pd.Timestamp("2026-05-04 05:30:00")
        for day_offset in range(3):
            base_dt = d0 + pd.Timedelta(days=day_offset)
            for s in [111, 120, 129]:
                for slot in range(10, 86):
                    dt = base_dt + pd.Timedelta(minutes=(slot - 10) * 15)
                    records.append({
                        "datetime": dt,
                        "station_code": s,
                        "passengers": 200.0 + (s - 110) * 10 + slot * 2
                    })
        cls.df = pd.DataFrame(records)
        cls.baseline = MetroBaselineModel()
        cls.baseline.fit(cls.df)
        cls.builder = MetroFeatureBuilder(baseline_model=cls.baseline)

    def test_fit_and_predict_monotonic_quantiles(self):
        # Обучаем тестовую модель на горизонтах 15 и 30 мин
        model = MultiHorizonFlowModel(
            baseline_model=self.baseline,
            feature_builder=self.builder,
            horizons=[15, 30]
        )
        model.fit(self.df, num_boost_round=10)

        self.assertIn(15, model.models)
        self.assertIn(30, model.models)
        for H in [15, 30]:
            for q in [0.10, 0.50, 0.90]:
                self.assertIn(q, model.models[H])

        # Проверяем инференс и свойство монотонности
        inf_features = self.builder.build_inference_features(
            current_time=pd.Timestamp("2026-05-05 14:00:00"),
            history_pax_df=self.df,
            horizon_min=15
        )
        p10, p50, p90 = model.predict_horizon(inf_features, horizon_min=15)

        self.assertEqual(len(p10), 19)
        self.assertEqual(len(p50), 19)
        self.assertEqual(len(p90), 19)

        # Неотрицательность
        self.assertTrue((p10 >= 0.0).all())
        self.assertTrue((p50 >= 0.0).all())
        self.assertTrue((p90 >= 0.0).all())

        # Монотонность: P10 <= P50 <= P90
        self.assertTrue((p10 <= p50 + 1e-5).all(), "Нарушение: P10 > P50")
        self.assertTrue((p50 <= p90 + 1e-5).all(), "Нарушение: P50 > P90")

    def test_save_and_load(self):
        model = MultiHorizonFlowModel(
            baseline_model=self.baseline,
            feature_builder=self.builder,
            horizons=[15]
        )
        model.fit(self.df, num_boost_round=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file = os.path.join(tmpdir, "test_models.joblib")
            model.save(save_file)
            self.assertTrue(os.path.exists(save_file))

            loaded = MultiHorizonFlowModel(
                baseline_model=self.baseline,
                feature_builder=self.builder
            ).load(save_file)
            self.assertIn(15, loaded.models)
            self.assertEqual(loaded.horizons, [15])


if __name__ == "__main__":
    unittest.main()
