"""
Интерфейсный модуль промышленного прогнозатора пассажиропотока («МетроПульс-1»).
Задача F8:
Реализация production API-контракта для интеграции с:
- Решающим диспетчерским модулем (src/dispatch/decision_engine.py Федора)
- Бэкендом FastAPI (src/api/ Кирилла)

Класс: MetroFlowPredictor
Метод: predict(current_time, history_pax_df, external_context=None) -> pd.DataFrame
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any, List, Union
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.stations import STATIONS_LINE_1
from src.data.operating_hours import operating_mask
from src.models.baseline import MetroBaselineModel
from src.models.feature_builder import MetroFeatureBuilder, STATION_METADATA, FEATURE_COLUMNS
from src.models.train_multi_horizon import MultiHorizonFlowModel, SUPPORTED_HORIZONS
from src.models.explainability import MetroExplainabilityEngine


class MetroFlowPredictor:
    """
    Промышленный сервис прогнозирования входящих пассажиропотоков станций Линии 1.
    Интегрирует детерминированный медианный профиль, бленд LightGBM+Ridge и модуль Semantic XAI.
    """

    def __init__(
        self,
        models_dir: Optional[str] = None,
        z_anomaly_threshold: float = 2.0,
        pct_anomaly_threshold: float = 0.15,
        abs_anomaly_threshold: float = 100.0,
        platform_capacity: float = 1200.0
    ):
        self.models_dir = models_dir or self._resolve_models_dir()
        self.z_threshold = z_anomaly_threshold
        self.pct_threshold = pct_anomaly_threshold
        self.abs_threshold = abs_anomaly_threshold
        self.platform_capacity = platform_capacity

        self.baseline_model = MetroBaselineModel()
        self.mh_model: Optional[MultiHorizonFlowModel] = None
        self.feature_builder: Optional[MetroFeatureBuilder] = None
        self.explain_engine = MetroExplainabilityEngine(feature_names=FEATURE_COLUMNS)

        self._load_models()

    def _resolve_models_dir(self) -> str:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(base_dir, "models")

    def _load_models(self):
        """Загружает базовый профиль и обученные мультигоризонтные квантильные модели."""
        base_path = os.path.join(self.models_dir, "baseline_profile.parquet")
        if os.path.exists(base_path):
            self.baseline_model.load(base_path)
            
        self.feature_builder = MetroFeatureBuilder(baseline_model=self.baseline_model)
        
        mh_path = os.path.join(self.models_dir, "multi_horizon_models.joblib")
        if os.path.exists(mh_path):
            self.mh_model = MultiHorizonFlowModel(
                baseline_model=self.baseline_model,
                feature_builder=self.feature_builder
            ).load(mh_path)
        else:
            self.mh_model = None

    def predict(
        self,
        current_time: Union[str, pd.Timestamp],
        history_pax_df: Optional[pd.DataFrame] = None,
        external_context: Optional[Dict[str, Any]] = None,
        horizons: Optional[List[int]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Возвращает прогноз по всем 19 станциям Линии 1 на горизонты 15, 30, 60, 120 минут.

        Параметры:
        - current_time: текущий операционный момент времени (Timestamp или str)
        - history_pax_df: история фактических входов (колонки: datetime, station_code, passengers)
        - external_context: словарь внешних факторов (погода, дорожные события, происшествия)
        - horizons: список горизонтов (по умолчанию: [15, 30, 60, 120])

        Возвращаемый DataFrame:
        - station_code: int
        - station_name: str
        - horizon_min: int (15, 30, 60, 120)
        - target_datetime: pd.Timestamp
        - base_pax: float (медианная норма графика)
        - pred_p50: float (ожидаемый поток)
        - pred_p90: float (верхняя граница риска)
        - pred_p10: float (нижняя граница)
        - is_anomaly: bool (флаг критического отклонения)
        - anomaly_score: float (робастный Z-score отклонения)
        """
        current_dt = pd.to_datetime(current_time)
        target_horizons = horizons or SUPPORTED_HORIZONS
        all_stations = sorted(list(STATION_METADATA.keys()))

        if history_pax_df is None and "history_df" in kwargs:
            history_pax_df = kwargs["history_df"]

        # Если истории нет, создаем синтетическую пустую структуру
        if history_pax_df is None or len(history_pax_df) == 0:
            history_pax_df = pd.DataFrame(columns=["datetime", "station_code", "passengers"])

        forecast_frames = []

        for H in target_horizons:
            target_dt = current_dt + pd.Timedelta(minutes=H)
            is_target_operating = bool(operating_mask(pd.Series([target_dt])).iloc[0])

            # 1. Формируем признаки для инференса
            inf_features = self.feature_builder.build_inference_features(
                current_time=current_dt,
                history_pax_df=history_pax_df,
                horizon_min=H
            )

            # 2. Получаем базовый профиль
            base_pax_arr = inf_features["base_pax_target"].values
            base_mad_arr = inf_features["base_mad_target"].values

            # 3. Вычисляем квантильные прогнозы ML-модели
            if self.mh_model is not None and H in self.mh_model.models:
                p10_arr, p50_arr, p90_arr = self.mh_model.predict_horizon(inf_features, horizon_min=H)
            else:
                # Fallback на базовый профиль с эвристическими коридорами
                p50_arr = base_pax_arr.copy()
                p10_arr = np.maximum(0.0, base_pax_arr - 1.28 * base_mad_arr * 1.4826)
                p90_arr = base_pax_arr + 1.28 * base_mad_arr * 1.4826

            # Если целевое время выпадает на нерабочие ночные часы метро (00:30-05:30), поток нулевой
            if not is_target_operating:
                base_pax_arr = np.zeros_like(base_pax_arr)
                p10_arr = np.zeros_like(p10_arr)
                p50_arr = np.zeros_like(p50_arr)
                p90_arr = np.zeros_like(p90_arr)

            # 4. Расчет робастного Z-score (anomaly_score) и флага аномалии
            # MAD (median absolute deviation) -> sigma_est = 1.4826 * MAD
            robust_sigma = np.maximum(base_mad_arr * 1.4826, 15.0)
            anomaly_score_arr = np.where(
                is_target_operating,
                (p50_arr - base_pax_arr) / robust_sigma,
                0.0
            )

            # Логика детекции аномалии (согласовано с регламентами ЦУП и WORK_PLAN.md §3.2 D3):
            # 1. Значительное статистическое отклонение от нормы (Z >= 2.0)
            # 2. Существенный относительный рост (+15%) при абсолютном приросте > 100 чел И умеренно повышенном Z >= 1.5
            # 3. Риск перегруза платформы: P90 превышает емкость станции (не менее 125% нормы или лимита состава)
            capacity_limit = (
                np.maximum(base_pax_arr * 1.25, self.platform_capacity)
                if self.platform_capacity is not None
                else base_pax_arr * 1.25
            )
            is_anomaly_arr = (
                is_target_operating & (
                    (anomaly_score_arr >= self.z_threshold) |
                    (
                        (p50_arr > (base_pax_arr * (1.0 + self.pct_threshold))) & 
                        ((p50_arr - base_pax_arr) >= self.abs_threshold) & 
                        (anomaly_score_arr >= 1.5)
                    ) |
                    ((p90_arr >= capacity_limit) & (anomaly_score_arr >= 1.5))
                )
            )

            # 5. Сборка датафрейма для данного горизонта
            h_df = pd.DataFrame({
                "station_code": inf_features["station_code"].astype(int),
                "station_name": inf_features["station_name"].astype(str),
                "horizon_min": H,
                "target_datetime": target_dt,
                "base_pax": np.round(base_pax_arr, 1),
                "pred_p50": np.round(p50_arr, 1),
                "pred_p90": np.round(p90_arr, 1),
                "pred_p10": np.round(p10_arr, 1),
                "is_anomaly": is_anomaly_arr.astype(bool),
                "anomaly_score": np.round(anomaly_score_arr, 2),
                # Псевдонимы для совместимости с контрактами WORK_PLAN.md §4
                "base": np.round(base_pax_arr, 1),
                "p50": np.round(p50_arr, 1),
                "p90": np.round(p90_arr, 1),
                "p10": np.round(p10_arr, 1),
                "ts": target_dt,
                "station": inf_features["station_code"].astype(int)
            })
            forecast_frames.append(h_df)

        result_df = pd.concat(forecast_frames, axis=0).sort_values(
            ["horizon_min", "station_code"]
        ).reset_index(drop=True)

        return result_df

    def explain_station(
        self,
        current_time: Union[str, pd.Timestamp],
        station_code: int,
        horizon_min: int = 30,
        history_pax_df: Optional[pd.DataFrame] = None,
        external_context: Optional[Dict[str, Any]] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Генерирует семантическое объяснение прогноза (Semantic XAI) для конкретной станции.
        Возвращает структурированный словарь с top-k причинами на русском языке и долями влияния.
        """
        current_dt = pd.to_datetime(current_time)
        if history_pax_df is None:
            history_pax_df = pd.DataFrame(columns=["datetime", "station_code", "passengers"])

        inf_features = self.feature_builder.build_inference_features(
            current_time=current_dt,
            history_pax_df=history_pax_df,
            horizon_min=horizon_min
        )

        station_row = inf_features[inf_features["station_code"] == station_code]
        if station_row.empty:
            return {"station_code": station_code, "reasons": [], "summary": "Станция не найдена"}

        booster = None
        if self.mh_model is not None and horizon_min in self.mh_model.models:
            booster = self.mh_model.models[horizon_min][0.50]

        reasons = self.explain_engine.explain_prediction(
            booster=booster,
            features_row=station_row,
            external_context=external_context,
            top_k=top_k
        )
        summary = self.explain_engine.format_reasons_bullet(reasons)

        return {
            "station_code": station_code,
            "horizon_min": horizon_min,
            "reasons": reasons,
            "summary": summary
        }

    def to_dispatch_records(
        self,
        forecast_df: pd.DataFrame,
        history_pax_df: Optional[pd.DataFrame] = None,
        external_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Конвертирует результат прогноза в формат JSON-контракта между модулями (WORK_PLAN.md §4).
        Для аномальных станций автоматически подтягивает семантическое объяснение причин (Semantic XAI).
        """
        records = []
        for _, row in forecast_df.iterrows():
            ts_val = row["target_datetime"]
            ts_str = ts_val.isoformat() if hasattr(ts_val, "isoformat") else str(ts_val)
            is_anom = bool(row["is_anomaly"])
            st_code = int(row["station_code"])
            h_min = int(row["horizon_min"])

            reasons_list = []
            summary_text = "Штатный режим"
            if is_anom:
                exp = self.explain_station(
                    current_time=ts_val - pd.Timedelta(minutes=h_min),
                    station_code=st_code,
                    horizon_min=h_min,
                    history_pax_df=history_pax_df,
                    external_context=external_context
                )
                reasons_list = exp.get("reasons", [])
                summary_text = exp.get("summary", "")

            records.append({
                "ts": ts_str,
                "station": st_code,
                "station_name": str(row["station_name"]),
                "horizon_min": h_min,
                "base": float(row["base_pax"]),
                "p10": float(row["pred_p10"]),
                "p50": float(row["pred_p50"]),
                "p90": float(row["pred_p90"]),
                "is_anomaly": is_anom,
                "anomaly_score": float(row["anomaly_score"]),
                "reasons": reasons_list,
                "explanation_text": summary_text
            })
        return records


if __name__ == "__main__":
    from src.data.data_loader import MetroDataLoader

    loader = MetroDataLoader()
    df_raw = loader.load_15min_dataset()
    df_station = loader.get_station_aggregated_flow(df_raw)

    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ПРОИЗВОДСТВЕННОГО ИНТЕРФЕЙСА MetroFlowPredictor (F8)")
    print("=" * 70)

    predictor = MetroFlowPredictor()
    
    # Симулируем онлайн-вызов диспетчера: вечер предпраздничного дня (08 мая 2026, 16:30)
    sim_time = pd.Timestamp("2026-05-08 16:30:00")
    recent_history = df_station[
        (df_station["datetime"] <= sim_time) &
        (df_station["datetime"] >= sim_time - pd.Timedelta(hours=2))
    ]

    print(f"\nЗапрос прогноза на {sim_time} (история: {len(recent_history)} записей по станциям)...")
    forecast = predictor.predict(current_time=sim_time, history_pax_df=recent_history)

    print(f"Форма ответа: {forecast.shape} (ожидается: 19 станций * 4 горизонта = 76 строк)")
    print("\nПервые 8 строк прогноза (горизонт 15 мин):")
    print(forecast[forecast["horizon_min"] == 15].head(8).to_string(index=False))

    anomalies = forecast[forecast["is_anomaly"]]
    print(f"\nОбнаружено аномальных станций/горизонтов: {len(anomalies)}")
    if len(anomalies) > 0:
        print(anomalies[["station_name", "horizon_min", "base_pax", "pred_p50", "pred_p90", "anomaly_score"]].head(10).to_string(index=False))
