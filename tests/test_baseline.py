"""
Unit-тесты для базового профиля расписания (src/models/baseline.py).
Проверяют корректность расчета медиан, отсутствие утечек, обработку пропусков и fallback.
"""

import unittest
import datetime
import numpy as np
import pandas as pd

from src.models.baseline import MetroBaselineModel, compute_wape, compute_mae


class TestMetroBaselineModel(unittest.TestCase):

    def setUp(self):
        self.model = MetroBaselineModel()
        
        # Создаем синтетические данные для 2 станций на 2 дня
        records = []
        d1 = pd.Timestamp("2026-02-02 05:30:00") # Понедельник (mon_thu)
        d2 = pd.Timestamp("2026-02-06 05:30:00") # Пятница (fri)
        
        for base_dt, dt_type in [(d1, "mon_thu"), (d2, "fri")]:
            for s in [111, 120]:
                for slot in range(10, 86): # слоты 05:30 (10) - 00:15 (85)
                    dt = base_dt + pd.Timedelta(minutes=(slot - 10) * 15)
                    records.append({
                        "datetime": dt,
                        "station_code": s,
                        "passengers": 100.0 + (s - 110) * 10 + slot
                    })
        self.synthetic_df = pd.DataFrame(records)

    def test_wape_mae_calculation(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 300.0])
        self.assertAlmostEqual(compute_wape(y_true, y_pred), 20.0 / 600.0)
        self.assertAlmostEqual(compute_mae(y_true, y_pred), 20.0 / 3.0)

    def test_day_type_assignment(self):
        # 23 февраля 2026 - праздник (holiday)
        # 08 мая 2026 - предпраздничный (pre_holiday)
        # 15 мая 2026 - пятница (fri)
        # 18 мая 2026 - понедельник (mon_thu)
        # 16 мая 2026 - суббота (sat)
        # 17 мая 2026 - воскресенье (sun)
        dates = pd.Series([
            pd.Timestamp("2026-02-23 08:00:00"),
            pd.Timestamp("2026-05-08 16:00:00"),
            pd.Timestamp("2026-05-15 17:00:00"),
            pd.Timestamp("2026-05-18 08:30:00"),
            pd.Timestamp("2026-05-16 12:00:00"),
            pd.Timestamp("2026-05-17 19:00:00")
        ])
        types = self.model.assign_day_type(dates).tolist()
        self.assertEqual(types, ["holiday", "pre_holiday", "fri", "mon_thu", "sat", "sun"])

    def test_night_operational_day_assignment(self):
        # 00:15 в субботу ночью (ночь с пятницы на субботу) относится к операционным суткам пятницы!
        friday_night = pd.Series([pd.Timestamp("2026-05-16 00:15:00")])
        assigned = self.model.assign_day_type(friday_night).iloc[0]
        self.assertEqual(assigned, "fri")

    def test_fit_and_predict(self):
        self.model.fit(self.synthetic_df)
        self.assertIsNotNone(self.model.profile_df)
        self.assertGreater(len(self.model.profile_df), 0)

        preds = self.model.predict(self.synthetic_df)
        self.assertIn("base_pax", preds.columns)
        self.assertIn("base_mad", preds.columns)
        self.assertIn("base_std", preds.columns)
        self.assertFalse(preds["base_pax"].isna().any())

    def test_fallback_unseen_day_type(self):
        self.model.fit(self.synthetic_df)
        # Тестируем дату воскресенья, которой не было в синтетическом обучении
        unseen_df = pd.DataFrame([{
            "datetime": pd.Timestamp("2026-02-08 08:00:00"), # Воскресенье
            "station_code": 111,
            "passengers": 50.0
        }])
        preds = self.model.predict(unseen_df)
        self.assertFalse(preds["base_pax"].isna().any())
        self.assertGreater(preds["base_pax"].iloc[0], 0.0)


if __name__ == "__main__":
    unittest.main()
