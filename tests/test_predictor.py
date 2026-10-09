"""
Unit-тесты для интерфейса MetroFlowPredictor (src/models/predictor.py).
Проверяют соответствие API-контракту СППР Линии 1, обработку граничных случаев,
типы данных и логику детекции аномалий.
"""

import unittest
import numpy as np
import pandas as pd

from src.models.predictor import MetroFlowPredictor


class TestMetroFlowPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Загружаем реальный инстанс с сохраненными артефактами
        cls.predictor = MetroFlowPredictor()

    def test_predict_contract_shape_and_columns(self):
        current_time = pd.Timestamp("2026-05-08 16:30:00")
        
        # Симулируем 2 часа истории
        history = []
        for dt in pd.date_range("2026-05-08 14:30:00", current_time, freq="15min"):
            for code in range(111, 130):
                history.append({
                    "datetime": dt,
                    "station_code": code,
                    "passengers": 500.0
                })
        history_df = pd.DataFrame(history)

        forecast_df = self.predictor.predict(current_time, history_df=history_df)

        # 1. Размер: 19 станций * 4 горизонта = 76 строк
        self.assertEqual(len(forecast_df), 76)
        
        # 2. Обязательные колонки по контракту (§7 архитектурного плана)
        expected_cols = [
            "station_code", "station_name", "horizon_min", "target_datetime",
            "base_pax", "pred_p50", "pred_p90", "pred_p10", "is_anomaly", "anomaly_score"
        ]
        for col in expected_cols:
            self.assertIn(col, forecast_df.columns, f"Отсутствует обязательная колонка {col}")

        # 3. Проверка типов
        self.assertTrue(np.issubdtype(forecast_df["station_code"].dtype, np.integer))
        self.assertTrue(pd.api.types.is_string_dtype(forecast_df["station_name"]))
        self.assertTrue(np.issubdtype(forecast_df["horizon_min"].dtype, np.integer))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(forecast_df["target_datetime"]))
        self.assertEqual(forecast_df["is_anomaly"].dtype, bool)
        self.assertTrue(np.issubdtype(forecast_df["base_pax"].dtype, np.floating))
        self.assertTrue(np.issubdtype(forecast_df["pred_p50"].dtype, np.floating))

        # 4. Проверка горизонтов
        self.assertEqual(sorted(forecast_df["horizon_min"].unique()), [15, 30, 60, 120])

        # 5. Проверка монотонности квантилей: P10 <= P50 <= P90
        p10 = forecast_df["pred_p10"].values
        p50 = forecast_df["pred_p50"].values
        p90 = forecast_df["pred_p90"].values
        self.assertTrue((p10 <= p50 + 1e-3).all(), "Нарушена монотонность: P10 > P50")
        self.assertTrue((p50 <= p90 + 1e-3).all(), "Нарушена монотонность: P50 > P90")

    def test_empty_history_graceful_fallback(self):
        # При отсутствии истории предиктор обязан вернуть корректный датафрейм (fallback на baseline)
        forecast_df = self.predictor.predict(
            current_time="2026-05-08 10:00:00",
            history_pax_df=None
        )
        self.assertEqual(len(forecast_df), 76)
        self.assertFalse(forecast_df["base_pax"].isna().any())
        self.assertFalse(forecast_df["pred_p50"].isna().any())
        self.assertTrue((forecast_df["pred_p50"] >= 0.0).all())

    def test_night_operating_hours_cutoff(self):
        # 01:00 ночи - метро закрыто, прогнозируемый пассажиропоток должен быть 0.0
        forecast_df = self.predictor.predict(
            current_time="2026-05-08 01:00:00",
            history_pax_df=None,
            horizons=[15]
        )
        self.assertEqual(len(forecast_df), 19)
        self.assertTrue((forecast_df["pred_p50"] == 0.0).all())
        self.assertTrue((forecast_df["base_pax"] == 0.0).all())
        self.assertFalse(forecast_df["is_anomaly"].any())

    def test_anomaly_flagging_on_massive_surge(self):
        # Симулируем искусственный мощный всплеск на станции 120 (Площадь Восстания)
        current_time = pd.Timestamp("2026-05-08 17:00:00")
        history = []
        for dt in pd.date_range("2026-05-08 15:00:00", current_time, freq="15min"):
            for code in range(111, 130):
                # На станции 120 задаем гигантский приток (например, 5000 чел)
                pax_val = 5000.0 if code == 120 else 300.0
                history.append({
                    "datetime": dt,
                    "station_code": code,
                    "passengers": pax_val
                })
        history_df = pd.DataFrame(history)

        forecast_df = self.predictor.predict(current_time, history_df=history_df, horizons=[15])
        st120 = forecast_df[forecast_df["station_code"] == 120].iloc[0]

        # Для ст. 120 флаг аномалии обязан сработать
        self.assertTrue(st120["is_anomaly"], "Аномальный всплеск на станции 120 не был обнаружен!")
        self.assertGreater(st120["anomaly_score"], 2.0, "Z-score всплеска ниже ожидаемого порога!")

    def test_no_false_alarm_on_normal_weekday(self):
        # На регулярном буднем дне (среда 08:30) при нормальном пассажиропотоке система обязана молчать (контроль: система молчит)
        current_time = pd.Timestamp("2026-05-13 08:30:00")
        all_codes = list(range(111, 130))
        history = []
        for dt in pd.date_range("2026-05-13 06:30:00", current_time, freq="15min"):
            target_df = pd.DataFrame({
                "station_code": all_codes,
                "datetime": [dt] * len(all_codes)
            })
            base_df = self.predictor.baseline_model.predict(target_df)
            for _, r in base_df.iterrows():
                history.append({
                    "datetime": dt,
                    "station_code": int(r["station_code"]),
                    "passengers": float(r["base_pax"]) # штатный поток по графику
                })
        history_df = pd.DataFrame(history)

        forecast_df = self.predictor.predict(current_time, history_df=history_df, horizons=[15, 30])
        # При штатном потоке ни одна станция не должна быть помечена как аномальная
        anomalous_stations = forecast_df[forecast_df["is_anomaly"]]
        self.assertEqual(len(anomalous_stations), 0, f"Ложные срабатывания на штатном расписании: {anomalous_stations['station_name'].tolist()}")

    def test_vestibule_level_history_auto_aggregated(self):
        # Проверяем, что при подаче телеметрии по 24 вестибюлям предиктор не путает вестибюли с лагами во времени
        current_time = pd.Timestamp("2026-05-08 16:30:00")
        history = [
            {"datetime": current_time, "station_code": 111, "vestibule": "пр. Ветеранов-1", "passengers": 400.0},
            {"datetime": current_time, "station_code": 111, "vestibule": "пр. Ветеранов-2", "passengers": 300.0},
            {"datetime": current_time - pd.Timedelta(minutes=15), "station_code": 111, "vestibule": "пр. Ветеранов-1", "passengers": 380.0},
            {"datetime": current_time - pd.Timedelta(minutes=15), "station_code": 111, "vestibule": "пр. Ветеранов-2", "passengers": 310.0},
        ]
        history_df = pd.DataFrame(history)

        forecast_df = self.predictor.predict(current_time, history_df=history_df, horizons=[15])
        self.assertEqual(len(forecast_df), 19)
        st111 = forecast_df[forecast_df["station_code"] == 111].iloc[0]
        # Проверяем, что прогноз построен корректно без падений
        self.assertGreater(st111["pred_p50"], 0.0)

    def test_stale_telemetry_safe_handling(self):
        # Данные 2-часовой давности не должны маскироваться под свежее состояние момента T
        current_time = pd.Timestamp("2026-05-08 16:30:00")
        stale_history = pd.DataFrame([
            {"datetime": pd.Timestamp("2026-05-08 14:00:00"), "station_code": 111, "passengers": 2500.0}
        ])

        forecast_df = self.predictor.predict(current_time, history_df=stale_history, horizons=[15])
        st111 = forecast_df[forecast_df["station_code"] == 111].iloc[0]
        # Устаревшие данные должны быть сброшены, Z-score не должен раздуваться
        self.assertLess(st111["anomaly_score"], 2.0, "Устаревшие данные вызвали ложную тревогу!")

    def test_dispatch_contract_records_format(self):
        # Проверяем формат to_dispatch_records по контракту §4 WORK_PLAN.md
        current_time = pd.Timestamp("2026-05-08 16:30:00")
        forecast_df = self.predictor.predict(current_time, horizons=[15])
        records = self.predictor.to_dispatch_records(forecast_df)

        self.assertEqual(len(records), 19)
        sample = records[0]
        for key in ["ts", "station", "station_name", "horizon_min", "base", "p10", "p50", "p90", "is_anomaly", "anomaly_score"]:
            self.assertIn(key, sample, f"Отсутствует обязательный ключ {key} в JSON-контракте")
        self.assertIsInstance(sample["ts"], str)
        self.assertIsInstance(sample["station"], int)
        self.assertIsInstance(sample["is_anomaly"], bool)


if __name__ == "__main__":
    unittest.main()
