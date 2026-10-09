"""
Модель базового расписания пассажиропотока Линии 1 («МетроПульс-1»).
Задача F1: медианный профиль по (станция × тип дня × 15-мин слот).

Типы дней (6 классов):
- 'mon_thu': понедельник - четверг (стандартный рабочий день)
- 'fri': пятница (смещение вечернего пика на 16:00, ранний отток)
- 'sat': суббота (сглаженный полуденный профиль)
- 'sun': воскресенье (вечерний приток приезжающих)
- 'holiday': официальные праздники РФ
- 'pre_holiday': предпраздничные сокращенные дни (сдвиг пика на 15:00-16:00)

Фильтрация: пассажирские часы 05:30-00:30 (ночные технологические часы 00:30-05:15 отсекаются).
Ориентир качества (LOMO CV): WAPE ≈ 13.0% - 13.7%.
"""

from __future__ import annotations

import os
import json
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.operating_hours import filter_operating_hours, operating_mask


DAY_TYPE_ORDER = ["mon_thu", "fri", "sat", "sun", "holiday", "pre_holiday"]

FALLBACK_MAPPING = {
    "holiday": "sun",
    "pre_holiday": "fri",
}


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Вычисляет взвешенную абсолютную процентную ошибку (WAPE = sum(|y - p|) / sum(y))."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(y_true)
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Вычисляет среднюю абсолютную ошибку (MAE)."""
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


class MetroBaselineModel:
    """
    Базовая модель медианного профиля станций Петербургского метрополитена.
    Обеспечивает детерминированный прогноз пассажиропотока без оверфиттинга.
    """

    def __init__(self, calendar_path: Optional[str] = None):
        self.calendar_path = calendar_path or self._resolve_calendar_path()
        self.calendar_map: Dict[str, int] = {}
        self._load_calendar()
        
        # Таблицы профилей
        self.profile_df: Optional[pd.DataFrame] = None
        self.global_slot_medians: Optional[pd.Series] = None
        self.station_slot_medians: Optional[pd.DataFrame] = None

    def _resolve_calendar_path(self) -> str:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(base_dir, "data", "calendar_2026.csv")

    def _load_calendar(self):
        """Загружает календарь производственных дней РФ (isdayoff)."""
        if os.path.exists(self.calendar_path):
            cal_df = pd.read_csv(self.calendar_path)
            cal_df["day_str"] = pd.to_datetime(cal_df["day"]).dt.strftime("%Y-%m-%d")
            self.calendar_map = dict(zip(cal_df["day_str"], cal_df["code"].astype(int)))

    def assign_day_type(self, dt_series: pd.Series) -> pd.Series:
        """
        Классифицирует даты на 6 типов:
        'holiday', 'pre_holiday', 'fri', 'mon_thu', 'sat', 'sun'.
        Учитывает операционные сутки метро: ночные часы (00:00-02:45) относятся к предыдущему дню!
        """
        dt = pd.to_datetime(dt_series)
        # Операционные сутки со сдвигом на 3 часа
        op_date = (dt - pd.Timedelta(hours=3)).dt.normalize()
        dow = op_date.dt.dayofweek
        op_date_str = op_date.dt.strftime("%Y-%m-%d")

        # Код календаря: 0 - рабочий, 1 - выходной/праздник, 2 - сокращенный рабочий
        code_series = op_date_str.map(self.calendar_map).fillna(0).astype(int)

        is_off = (code_series == 1)
        is_pre = (code_series == 2)

        # Логика типов:
        # 1. Праздник, если официальный выходной в будни (Пн-Пт) или известные ключевые даты (09.05)
        # 2. Предпраздничный, если code == 2
        # 3. Пятница, если dow == 4
        # 4. Понедельник-Четверг, если dow < 4
        # 5. Суббота, если dow == 5
        # 6. Воскресенье, если dow == 6
        conditions = [
            (is_off & (dow < 5)) | (op_date_str == "2026-05-09"),
            is_pre,
            dow == 4,
            dow < 4,
            dow == 5
        ]
        choices = ["holiday", "pre_holiday", "fri", "mon_thu", "sat"]
        day_type = np.select(conditions, choices, default="sun")
        return pd.Series(day_type, index=dt_series.index, name="day_type")

    @staticmethod
    def get_slot_index(dt_series: pd.Series) -> pd.Series:
        """Возвращает 15-минутный слот суток от полуночи (0..95: hour * 4 + minute // 15)."""
        dt = pd.to_datetime(dt_series)
        return pd.Series(dt.dt.hour * 4 + (dt.dt.minute // 15), index=dt_series.index, name="slot_index")

    def fit(self, df: pd.DataFrame, target_col: str = "passengers") -> MetroBaselineModel:
        """
        Строит медианные профили по (station_code, day_type, slot_index).
        Автоматически фильтрует пассажирские часы (05:30-00:30).
        """
        df_clean = filter_operating_hours(df, dt_col="datetime").copy()
        
        if "day_type" not in df_clean.columns:
            df_clean["day_type"] = self.assign_day_type(df_clean["datetime"])
        if "slot_index" not in df_clean.columns:
            df_clean["slot_index"] = self.get_slot_index(df_clean["datetime"])

        group_cols = ["station_code", "day_type", "slot_index"]
        
        # Агрегация статистик: медиана, среднее, MAD, std, P10, P90
        def calc_mad(x):
            med = np.median(x)
            return float(np.median(np.abs(x - med)))

        agg_df = df_clean.groupby(group_cols)[target_col].agg(
            base_pax="median",
            mean_pax="mean",
            std_pax="std",
            mad_pax=calc_mad,
            p10_pax=lambda x: np.percentile(x, 10),
            p90_pax=lambda x: np.percentile(x, 90),
            count="count"
        ).reset_index()

        agg_df["std_pax"] = agg_df["std_pax"].fillna(0.0)
        agg_df["mad_pax"] = agg_df["mad_pax"].fillna(0.0)
        self.profile_df = agg_df

        # Fallback профили (по станции и слоту без учета дня, и глобальный по слоту)
        self.station_slot_medians = df_clean.groupby(["station_code", "slot_index"])[target_col].median().rename("base_station_slot").reset_index()
        self.global_slot_medians = df_clean.groupby("slot_index")[target_col].median().rename("base_global_slot")

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет базовый профиль к переданному DataFrame.
        Возвращает DataFrame с колонками base_pax, base_mad, base_std.
        """
        if self.profile_df is None:
            raise ValueError("Модель не обучена! Вызовите fit() перед predict().")

        orig_index = df.index
        res = df.copy().reset_index(drop=True)
        if "day_type" not in res.columns:
            res["day_type"] = self.assign_day_type(res["datetime"])
        if "slot_index" not in res.columns:
            res["slot_index"] = self.get_slot_index(res["datetime"])

        join_cols = ["station_code", "day_type", "slot_index"]
        merged = res.merge(self.profile_df, on=join_cols, how="left")

        # Обработка пропусков через Fallback (например, holiday -> sun, pre_holiday -> fri)
        missing_mask = merged["base_pax"].isna()
        if missing_mask.any():
            fallback_res = res.loc[missing_mask].copy()
            fallback_res["day_type"] = fallback_res["day_type"].map(lambda x: FALLBACK_MAPPING.get(x, x))
            fb_merged = fallback_res.merge(self.profile_df, on=join_cols, how="left")
            
            for col in ["base_pax", "mean_pax", "std_pax", "mad_pax", "p10_pax", "p90_pax"]:
                merged.loc[missing_mask, col] = fb_merged[col].values

        # Вторичный fallback на station_slot_medians
        still_missing = merged["base_pax"].isna()
        if still_missing.any():
            if self.station_slot_medians is not None:
                st_fb = res.loc[still_missing, ["station_code", "slot_index"]].merge(
                    self.station_slot_medians, on=["station_code", "slot_index"], how="left"
                )
                merged.loc[still_missing, "base_pax"] = st_fb["base_station_slot"].values
            else:
                merged.loc[still_missing, "base_pax"] = 0.0

        # Третичный fallback на global_slot_medians
        final_missing = merged["base_pax"].isna()
        if final_missing.any():
            if self.global_slot_medians is not None:
                gl_fb = res.loc[final_missing, "slot_index"].map(self.global_slot_medians).fillna(0.0)
                merged.loc[final_missing, "base_pax"] = gl_fb.values
            else:
                merged.loc[final_missing, "base_pax"] = 0.0

        # Значения по умолчанию для mad и std при пропусках (ночные слоты или непассажирские часы)
        merged["base_pax"] = merged["base_pax"].fillna(0.0)
        merged["mad_pax"] = merged["mad_pax"].fillna(np.maximum(merged["base_pax"] * 0.15, 5.0))
        merged["std_pax"] = merged["std_pax"].fillna(np.maximum(merged["base_pax"] * 0.20, 10.0))

        res["base_pax"] = merged["base_pax"].values
        res["base_mad"] = merged["mad_pax"].values
        res["base_std"] = merged["std_pax"].values
        res.index = orig_index
        return res

    def evaluate_lomo(
        self,
        df: pd.DataFrame,
        target_col: str = "passengers",
        months: Tuple[int, ...] = (2, 5, 7, 9)
    ) -> Tuple[Dict[int, float], float, pd.DataFrame]:
        """
        Честная 4-фолдовая Leave-One-Month-Out (LOMO) валидация:
        Каждый месяц исключается из обучения целиком и валидируется на медиане остальных 3 месяцев.
        """
        df_oper = filter_operating_hours(df, dt_col="datetime").copy()
        df_oper["month"] = pd.to_datetime(df_oper["datetime"]).dt.month
        df_oper["day_type"] = self.assign_day_type(df_oper["datetime"])
        df_oper["slot_index"] = self.get_slot_index(df_oper["datetime"])

        month_wapes = {}
        month_maes = {}
        all_preds = []

        for val_m in months:
            train_df = df_oper[df_oper["month"] != val_m].copy()
            val_df = df_oper[df_oper["month"] == val_m].copy()

            fold_model = MetroBaselineModel(calendar_path=self.calendar_path)
            fold_model.fit(train_df, target_col=target_col)
            pred_val = fold_model.predict(val_df)

            wape_score = compute_wape(pred_val[target_col].values, pred_val["base_pax"].values)
            mae_score = compute_mae(pred_val[target_col].values, pred_val["base_pax"].values)

            month_wapes[val_m] = wape_score
            month_maes[val_m] = mae_score
            all_preds.append(pred_val)

        mean_wape = float(np.mean(list(month_wapes.values())))
        combined_preds = pd.concat(all_preds, axis=0).reset_index(drop=True)
        return month_wapes, mean_wape, combined_preds

    def save(self, filepath: str):
        """Сохраняет обученный профиль в parquet/json."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        if filepath.endswith(".parquet"):
            self.profile_df.to_parquet(filepath, index=False)
        else:
            self.profile_df.to_json(filepath, orient="records", date_format="iso")

    def load(self, filepath: str) -> MetroBaselineModel:
        """Загружает профиль из файла."""
        if filepath.endswith(".parquet"):
            self.profile_df = pd.read_parquet(filepath)
        else:
            self.profile_df = pd.read_json(filepath)
        if self.profile_df is not None:
            self.station_slot_medians = self.profile_df.groupby(["station_code", "slot_index"])["base_pax"].median().rename("base_station_slot").reset_index()
            self.global_slot_medians = self.profile_df.groupby("slot_index")["base_pax"].median()
        return self


if __name__ == "__main__":
    from src.data.data_loader import MetroDataLoader
    
    loader = MetroDataLoader()
    df_raw = loader.load_15min_dataset()
    df_station = loader.get_station_aggregated_flow(df_raw)

    print("=" * 60)
    print("ВЫПОЛНЕНИЕ ЗАДАЧИ F1: МЕДИАННЫЙ ПРОФИЛЬ БЕЙЗЛАЙНА (LOMO)")
    print("=" * 60)
    
    model = MetroBaselineModel()
    month_wapes, mean_wape, preds_df = model.evaluate_lomo(df_station)
    
    month_names = {2: "Февраль (Зима)", 5: "Май (Праздники)", 7: "Июль (Лето)", 9: "Сентябрь (Осень)"}
    print("\nРезультаты Leave-One-Month-Out (LOMO) кросс-валидации:")
    print("-" * 50)
    for m in sorted(month_wapes.keys()):
        print(f"  Фолд [{month_names.get(m, str(m))}]: WAPE = {month_wapes[m]*100:.2f}%")
    print("-" * 50)
    print(f"  СВЕДЕННЫЙ WAPE: {mean_wape*100:.2f}%\n")

    # Обучаем финальный профиль на всех доступных 4 месяцах
    model.fit(df_station)
    save_path = os.path.join(loader.data_root, "..", "models", "baseline_profile.parquet")
    model.save(save_path)
    print(f"Базовый профиль сохранен в: {save_path} (всего слотов: {len(model.profile_df)})")
