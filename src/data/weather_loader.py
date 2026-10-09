"""
Модуль загрузки и обработки погодных данных Санкт-Петербурга через Open-Meteo Archive API.
Интерполирует почасовые метеоданные на 15-минутные интервалы и кэширует в Parquet.
"""

from typing import Optional
import os
import requests
import pandas as pd
import numpy as np


class WeatherLoader:
    """
    Загрузчик метеорологических факторов для Санкт-Петербурга (Линия 1).
    Координаты: 59.9343 N, 30.3351 E.
    """
    
    def __init__(
        self,
        cache_path: Optional[str] = None,
        lat: float = 59.9343,
        lon: float = 30.3351
    ):
        self.lat = lat
        self.lon = lon
        if cache_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.cache_path = os.path.join(base_dir, "data", "weather_spb_2026.parquet")
        else:
            self.cache_path = cache_path

    def fetch_weather_period(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Скачивает почасовую погоду из Open-Meteo Archive API."""
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={self.lat}&longitude={self.lon}&"
            f"start_date={start_date}&end_date={end_date}&"
            f"hourly=temperature_2m,apparent_temperature,precipitation,rain,snowfall,"
            f"wind_speed_10m,wind_gusts_10m,weather_code&"
            f"timezone=Europe/Moscow"
        )
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        hourly = data["hourly"]
        df = pd.DataFrame({
            "datetime": pd.to_datetime(hourly["time"]),
            "temperature": hourly["temperature_2m"],
            "apparent_temperature": hourly.get("apparent_temperature", hourly["temperature_2m"]),
            "precipitation": hourly["precipitation"],
            "rain": hourly["rain"],
            "snowfall": hourly["snowfall"],
            "wind_speed": hourly["wind_speed_10m"],
            "wind_gusts": hourly["wind_gusts_10m"],
            "weather_code": hourly["weather_code"],
        })
        return df

    def get_interpolated_15min_weather(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Возвращает 15-минутный временной ряд погоды для всех анализируемых периодов (2026).
        Если кэш существует и force_reload=False, загружает из parquet.
        """
        if os.path.exists(self.cache_path) and not force_reload:
            try:
                return pd.read_parquet(self.cache_path)
            except Exception:
                pass
                
        # Скачиваем 4 ключевых периода (февраль, май, июль, сентябрь 2026)
        periods = [
            ("2026-02-01", "2026-02-28"),
            ("2026-05-01", "2026-05-31"),
            ("2026-07-01", "2026-07-31"),
            ("2026-09-01", "2026-09-30"),
        ]
        
        frames = []
        for start_dt, end_dt in periods:
            df_part = self.fetch_weather_period(start_dt, end_dt)
            frames.append(df_part)
            
        hourly_df = pd.concat(frames, ignore_index=True)
        hourly_df = hourly_df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        
        # Интерполяция на 15-минутную сетку
        # Создаем полную 15-минутную сетку для каждого месяца
        fifteen_min_frames = []
        for start_dt, end_dt in periods:
            month_mask = (hourly_df["datetime"] >= start_dt) & (hourly_df["datetime"] <= f"{end_dt} 23:59:59")
            sub_h = hourly_df[month_mask].copy()
            sub_h = sub_h.set_index("datetime")
            
            # Resample до 15 мин
            full_idx = pd.date_range(start=f"{start_dt} 00:00:00", end=f"{end_dt} 23:45:00", freq="15min")
            sub_15 = sub_h.reindex(full_idx)
            
            # Числовые признаки интерполируем непрерывно
            num_cols = ["temperature", "apparent_temperature", "wind_speed", "wind_gusts"]
            sub_15[num_cols] = sub_15[num_cols].interpolate(method="time")
            
            # Осадки распределяем равномерно по 4 интервалам часа
            sub_15["precipitation"] = sub_15["precipitation"].ffill() / 4.0
            sub_15["rain"] = sub_15["rain"].ffill() / 4.0
            sub_15["snowfall"] = sub_15["snowfall"].ffill() / 4.0
            
            # Код погоды ffill
            sub_15["weather_code"] = sub_15["weather_code"].ffill().bfill().astype(int)
            
            sub_15 = sub_15.reset_index().rename(columns={"index": "datetime"})
            fifteen_min_frames.append(sub_15)
            
        full_weather = pd.concat(fifteen_min_frames, ignore_index=True)
        
        # Производные метео-фичи
        full_weather["is_rain"] = (full_weather["rain"] > 0.05).astype(int)
        full_weather["is_snow"] = (full_weather["snowfall"] > 0.02).astype(int)
        full_weather["is_heavy_rain"] = (full_weather["rain"] > 1.0).astype(int)
        full_weather["is_extreme_weather"] = (
            (full_weather["rain"] > 2.0) |
            (full_weather["snowfall"] > 0.5) |
            (full_weather["wind_gusts"] > 45.0) |
            (full_weather["temperature"] < -15.0)
        ).astype(int)
        
        # Сохранение в кэш
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        full_weather.to_parquet(self.cache_path, index=False)
        return full_weather
