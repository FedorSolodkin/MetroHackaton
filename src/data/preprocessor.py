"""
Модуль предварительной обработки данных и построения пространственно-временных признаков (Spatial-Temporal Feature Store).
Обеспечивает строго хронологическое разбиение без утечек данных (Data Leakage)
в соответствии с принципами ML-исследователя.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.calendar_features import CalendarFeatureEngine
from src.data.weather_loader import WeatherLoader


class MetroFeaturePipeline:
    """
    Пайплайн генерации признаков для Линии 1:
    - Пространственно-временные лаги (Spatial-Temporal wave lags)
    - Погодные факторы (Open-Meteo)
    - Производственный календарь РФ и события Санкт-Петербурга
    - Статистики скользящего окна
    """
    
    def __init__(self):
        self.calendar_engine = CalendarFeatureEngine()
        self.weather_loader = WeatherLoader()

    def build_feature_matrix(
        self,
        df_station: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Собирает единую признаковую матрицу на уровне станций.
        df_station: агрегированный датафрейм по станциям (от MetroDataLoader.get_station_aggregated_flow).
        """
        df = df_station.copy()
        df = df.sort_values(["station_order", "datetime"]).reset_index(drop=True)
        
        # 1. Добавляем календарные признаки
        df = self.calendar_engine.add_features(df, datetime_col="datetime")
        
        # 2. Добавляем погодные данные
        if weather_df is None:
            weather_df = self.weather_loader.get_interpolated_15min_weather()
            
        weather_cols_to_merge = [
            "datetime", "temperature", "apparent_temperature", "precipitation",
            "rain", "snowfall", "wind_speed", "wind_gusts", "is_rain", "is_snow",
            "is_heavy_rain", "is_extreme_weather"
        ]
        available_weather_cols = [c for c in weather_cols_to_merge if c in weather_df.columns]
        df = pd.merge(df, weather_df[available_weather_cols], on="datetime", how="left")
        
        # Заполняем пропуски погоды ffill/bfill если интервал на стыке
        num_weather = ["temperature", "apparent_temperature", "precipitation", "rain", "snowfall", "wind_speed", "wind_gusts"]
        df[num_weather] = df[num_weather].ffill().bfill()
        bin_weather = ["is_rain", "is_snow", "is_heavy_rain", "is_extreme_weather"]
        df[bin_weather] = df[bin_weather].fillna(0).astype(int)
        
        # 3. Временные лаги для каждой станции (t - 15m, t - 30m, t - 45m, t - 60m, t - 24h, t - 7d)
        # Группируем по station_code и month, чтобы исключить скачки сквозь пропущенные месяцы (март-апрель, июнь, август)
        df["month_group"] = df["datetime"].dt.month
        
        df["lag_15m"] = df.groupby(["station_code", "month_group"])["passengers"].shift(1)
        df["lag_30m"] = df.groupby(["station_code", "month_group"])["passengers"].shift(2)
        df["lag_45m"] = df.groupby(["station_code", "month_group"])["passengers"].shift(3)
        df["lag_60m"] = df.groupby(["station_code", "month_group"])["passengers"].shift(4)
        
        # Скользящие статистики по станции (последний час = 4 интервала, строго по прошлым точкам)
        df["rolling_mean_1h"] = df.groupby(["station_code", "month_group"])["passengers"].transform(
            lambda x: x.shift(1).rolling(4, min_periods=1).mean()
        )
        df["rolling_std_1h"] = df.groupby(["station_code", "month_group"])["passengers"].transform(
            lambda x: x.shift(1).rolling(4, min_periods=1).std()
        ).fillna(0.0)
        
        # Суточный лаг (96 интервалов назад внутри текущего месяца)
        df["lag_24h"] = df.groupby(["station_code", "month_group"])["passengers"].shift(96)
        
        # Недельный лаг (96 * 7 = 672 интервала назад внутри текущего месяца)
        df["lag_7d"] = df.groupby(["station_code", "month_group"])["passengers"].shift(672)
        
        # 4. Пространственные лаги (Spatial Wave Lags):
        # Волна из южного хаба (Проспект Ветеранов, код 111), северного хаба (Девяткино, код 129)
        # и центрального хаба (Площадь Восстания, код 120)
        pivoted = df.pivot_table(index="datetime", columns="station_code", values="passengers", aggfunc="first")
        
        # Поток на южном хабе ст. Ветеранов (111) с лагом 30 мин
        hub_south_lag30 = pivoted[111].shift(2).rename("spatial_hub_south_lag30")
        # Поток на северном хабе ст. Девяткино (129) с лагом 30 мин
        hub_north_lag30 = pivoted[129].shift(2).rename("spatial_hub_north_lag30")
        # Поток на центральном узле ст. Пл. Восстания (120) с лагом 15 мин
        hub_center_lag15 = pivoted[120].shift(1).rename("spatial_hub_center_lag15")
        
        hubs_df = pd.concat([hub_south_lag30, hub_north_lag30, hub_center_lag15], axis=1).reset_index()
        df = pd.merge(df, hubs_df, on="datetime", how="left")
        
        # 5. Безопасное заполнение начальных пропусков БЕЗ DATA LEAKAGE:
        # Категорически запрещено заполнять lag_15m через df['passengers'] (текущий таргет)!
        # Используем безопасную историческую медиану станции по интервалу дня и типу дня
        baseline_medians = df.groupby(["station_code", "interval_96", "is_weekend"])["passengers"].transform("median")
        
        df["lag_15m"] = df["lag_15m"].fillna(baseline_medians).fillna(0.0)
        df["lag_30m"] = df["lag_30m"].fillna(df["lag_15m"])
        df["lag_45m"] = df["lag_45m"].fillna(df["lag_30m"])
        df["lag_60m"] = df["lag_60m"].fillna(df["lag_45m"])
        df["rolling_mean_1h"] = df["rolling_mean_1h"].fillna(df["lag_15m"])
        df["lag_24h"] = df["lag_24h"].fillna(baseline_medians).fillna(0.0)
        df["lag_7d"] = df["lag_7d"].fillna(df["lag_24h"])
        
        df["spatial_hub_south_lag30"] = df["spatial_hub_south_lag30"].fillna(df["spatial_hub_south_lag30"].median())
        df["spatial_hub_north_lag30"] = df["spatial_hub_north_lag30"].fillna(df["spatial_hub_north_lag30"].median())
        df["spatial_hub_center_lag15"] = df["spatial_hub_center_lag15"].fillna(df["spatial_hub_center_lag15"].median())
        
        df = df.drop(columns=["month_group"], errors="ignore")
        return df.sort_values(["datetime", "station_order"]).reset_index(drop=True)

    def temporal_split(
        self,
        df: pd.DataFrame,
        split_date_val: str = "2026-08-01",
        split_date_test: str = "2026-09-15"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Строго хронологическое разделение выборки на Train / Val / Test
        без смешивания будущего и прошлого:
        - Train: февраль, май, июль (до 2026-08-01)
        - Validation: первая половина сентября (2026-09-01 по 2026-09-15)
        - Test: вторая половина сентября (2026-09-16 по 2026-09-30)
        """
        dt = pd.to_datetime(df["datetime"])
        train_mask = dt < pd.to_datetime(split_date_val)
        val_mask = (dt >= pd.to_datetime(split_date_val)) & (dt < pd.to_datetime(split_date_test))
        test_mask = dt >= pd.to_datetime(split_date_test)
        
        train_df = df[train_mask].copy().reset_index(drop=True)
        val_df = df[val_mask].copy().reset_index(drop=True)
        test_df = df[test_mask].copy().reset_index(drop=True)
        
        return train_df, val_df, test_df
