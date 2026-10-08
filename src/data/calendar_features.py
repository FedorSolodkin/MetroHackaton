"""
Модуль календарных и временных признаков для Петербургского метрополитена.
Учитывает производственный календарь РФ, переносы выходных, праздники СПб и цикличность суток.
"""

from typing import Dict, Any, Union
import datetime
import numpy as np
import pandas as pd
import holidays


class CalendarFeatureEngine:
    """
    Генератор календарных и временных фичей.
    Интегрирует производственный календарь России и городские события Санкт-Петербурга.
    """
    
    def __init__(self, years: list[int] = [2025, 2026]):
        self.years = years
        self.ru_holidays = holidays.Russia(years=years)
        
        # Специальные общегородские события Санкт-Петербурга (2026)
        self.spb_special_events = {
            datetime.date(2026, 5, 27): "День города (Санкт-Петербург)",
            datetime.date(2026, 6, 20): "Алые паруса (Главный выпускной)",
            datetime.date(2026, 6, 21): "Алые паруса (Ночной фестиваль)",
            datetime.date(2026, 7, 26): "День Военно-Морского Флота (Главный парад ВМФ)",
        }
        
        # Официальные сокращенные предпраздничные дни РФ (2026)
        self.pre_holidays = {
            datetime.date(2026, 2, 20),
            datetime.date(2026, 4, 30),
            datetime.date(2026, 5, 8),
            datetime.date(2026, 6, 11),
            datetime.date(2026, 11, 3),
            datetime.date(2026, 12, 31)
        }

    def get_day_metadata(self, date_val: Union[datetime.date, datetime.datetime, str]) -> Dict[str, Any]:
        """Возвращает метаданные для календарного дня."""
        if isinstance(date_val, str):
            dt = pd.to_datetime(date_val).date()
        elif isinstance(date_val, datetime.datetime):
            dt = date_val.date()
        else:
            dt = date_val
            
        is_official_holiday = dt in self.ru_holidays
        holiday_name = self.ru_holidays.get(dt, None)
        
        is_spb_event = dt in self.spb_special_events
        spb_event_name = self.spb_special_events.get(dt, None)
        
        is_pre_holiday = dt in self.pre_holidays
        weekday = dt.weekday() # 0 = Monday, 6 = Sunday
        is_weekend = weekday >= 5
        
        if is_official_holiday:
            day_type = "holiday"
        elif is_pre_holiday:
            day_type = "pre_holiday"
        elif is_weekend:
            day_type = "weekend"
        else:
            day_type = "workday"
            
        month = dt.month
        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        else:
            season = "autumn"
            
        return {
            "date": dt,
            "day_of_week": weekday,
            "day_name": ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"][weekday],
            "is_weekend": int(is_weekend),
            "is_holiday": int(is_official_holiday),
            "holiday_name": holiday_name,
            "is_pre_holiday": int(is_pre_holiday),
            "is_spb_event": int(is_spb_event),
            "spb_event_name": spb_event_name,
            "day_type": day_type,
            "season": season,
            "month": month,
            "day": dt.day
        }

    def add_features(self, df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
        """
        Обогащает датафрейм календарными, сезонными и циклическими признаками.
        df должен содержать столбец datetime_col с временной меткой.
        """
        df = df.copy()
        dt_series = pd.to_datetime(df[datetime_col])
        
        df["date"] = dt_series.dt.date
        df["hour"] = dt_series.dt.hour
        df["minute"] = dt_series.dt.minute
        
        # 15-минутный интервал дня (0..95)
        # Обратите внимание: в метрополитене сутки считаются с 03:00 до 02:45,
        # но стандартный интервал от полуночи: hour * 4 + minute // 15
        df["interval_96"] = df["hour"] * 4 + (df["minute"] // 15)
        
        # Циклические признаки времени суток (sin/cos)
        df["time_sin"] = np.sin(2 * np.pi * df["interval_96"] / 96.0)
        df["time_cos"] = np.cos(2 * np.pi * df["interval_96"] / 96.0)
        
        # Циклические признаки дня недели (sin/cos)
        df["day_of_week"] = dt_series.dt.weekday
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
        
        # Календарные флаги (календарная дата)
        unique_dates = df["date"].unique()
        meta_dict = {d: self.get_day_metadata(d) for d in unique_dates}
        meta_df = pd.DataFrame.from_dict(meta_dict, orient="index")
        
        for col in ["day_name", "is_weekend", "is_holiday", "is_pre_holiday", "is_spb_event", "day_type", "season", "month"]:
            df[col] = df["date"].map(meta_df[col])
            
        # Если передан operational_date (операционные сутки метро 03:00 - 02:45),
        # вычисляем операционные метаданные, чтобы ночные часы (00:00 - 02:45) не отрывались от смены
        if "operational_date" in df.columns:
            op_dates = df["operational_date"].apply(
                lambda d: pd.to_datetime(d).date() if not isinstance(d, datetime.date) else d
            )
            op_meta_dict = {d: self.get_day_metadata(d) for d in op_dates.unique()}
            op_meta_df = pd.DataFrame.from_dict(op_meta_dict, orient="index")
            for col in ["day_name", "is_weekend", "is_holiday", "is_pre_holiday", "is_spb_event", "day_type", "season", "month"]:
                df[f"op_{col}"] = op_dates.map(op_meta_df[col])
            # Основные категориальные признаки привязываем к операционным суткам
            df["day_name"] = df["op_day_name"]
            df["day_type"] = df["op_day_type"]
            df["is_weekend"] = df["op_is_weekend"]
            df["is_holiday"] = df["op_is_holiday"]
            df["is_pre_holiday"] = df["op_is_pre_holiday"]
            
        return df

CalendarFeatureEngineering = CalendarFeatureEngine
