"""
Модуль построения пространственно-временных признаков для мультигоризонтного прогнозирования («МетроПульс-1»).
Обеспечивает строгое соблюдение принципа Zero Data Leakage:
- При прогнозе на момент времени T для целевого горизонта t = T + H используются
  ТОЛЬКО данные, известные на момент T (или ранее).
- Единый пайплайн генерации признаков для оффлайн-обучения и онлайн-инференса (MetroFlowPredictor).
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.stations import STATIONS_LINE_1
from src.models.baseline import MetroBaselineModel

# Словарь метаданных станций
STATION_METADATA = {
    s["code"]: {
        "station_order": s.get("order", 0),
        "has_turnaround": int(s.get("has_turnaround", False)),
        "station_type": s.get("type", "intermediate"),
        "station_name": s.get("name", f"Станция {s['code']}")
    }
    for s in STATIONS_LINE_1
}

# Ключевые хабы Линии 1
HUB_SOUTH = 111   # Проспект Ветеранов (южный спальный терминал)
HUB_NORTH = 129   # Девяткино (северный спальный терминал + пригород)
HUB_CENTER = 120  # Площадь Восстания (главный пересадочный узел + Московский вокзал)

FEATURE_COLUMNS = [
    # Станционные признаки
    "station_code",
    "station_order",
    "has_turnaround",
    
    # Целевой временной и календарный контекст (t = T + H)
    "target_slot_index",
    "day_type_code",
    "dow",
    "is_weekend",
    "is_friday",
    "is_holiday",
    "is_pre_holiday",
    "target_time_sin",
    "target_time_cos",
    "target_dow_sin",
    "target_dow_cos",
    "base_pax_target",
    "base_mad_target",
    
    # Авторегрессия остатков и уровней на момент T (известно диспетчеру)
    "res_T",
    "res_T_15",
    "res_T_30",
    "res_T_45",
    "pax_T",
    "pax_T_15",
    "pax_T_30",
    "res_mean_1h",
    "res_std_1h",
    "res_trend",
    "pax_trend",
    "pax_ratio_base_T",
    "res_same_daytype_last",
    
    # Пространственные волны (Spatial Wave Lags) от ключевых узлов на момент T
    "south_hub_res_T",
    "north_hub_res_T",
    "center_hub_res_T",
    "south_hub_res_T_15",
    "north_hub_res_T_15",
    "center_hub_res_T_15",
]

CATEGORICAL_FEATURES = ["station_code", "day_type_code"]

DAY_TYPE_TO_CODE = {
    "mon_thu": 0,
    "fri": 1,
    "sat": 2,
    "sun": 3,
    "holiday": 4,
    "pre_holiday": 5
}


class MetroFeatureBuilder:
    """
    Генератор признаков для мультигоризонтных моделей LightGBM.
    Поддерживает:
    - Построение тренировочных датасетов по всей истории (с группировкой по месяцам)
    - Построение матрицы признаков на лету для инференса по текущему срезу истории
    """

    def __init__(self, baseline_model: Optional[MetroBaselineModel] = None):
        self.baseline_model = baseline_model or MetroBaselineModel()

    def _ensure_baseline_fitted(self, df_sample: pd.DataFrame):
        if self.baseline_model.profile_df is None:
            self.baseline_model.fit(df_sample)

    def build_training_dataset_for_horizon(
        self,
        df_station: pd.DataFrame,
        horizon_min: int
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Строит обучающий датасет для заданного горизонта horizon_min (минут).
        Гарантирует отсутствие утечек данных между месяцами.
        """
        self._ensure_baseline_fitted(df_station)
        
        # 1. Получаем базовый профиль и остатки
        df = self.baseline_model.predict(df_station).copy()
        df["residual"] = df["passengers"] - df["base_pax"]
        df["month"] = df["datetime"].dt.month

        # Станционные метаданные
        df["station_order"] = df["station_code"].map(lambda c: STATION_METADATA.get(c, {}).get("station_order", 0))
        df["has_turnaround"] = df["station_code"].map(lambda c: STATION_METADATA.get(c, {}).get("has_turnaround", 0))

        # Сортировка для корректных лагов
        df = df.sort_values(["station_code", "datetime"]).reset_index(drop=True)

        # 2. Лаг аналогичного дня того же типа внутри месяца
        df["res_same_daytype_last"] = df.groupby(
            ["station_code", "month", "day_type", "slot_index"]
        )["residual"].shift(1).fillna(0.0)

        # 3. Извлечение пространственных признаков хабов (Veteranov, Devyatkino, Vosstaniya)
        piv_res = df.pivot(index="datetime", columns="station_code", values="residual")
        hubs = pd.DataFrame(index=piv_res.index)
        hubs["datetime"] = piv_res.index
        hubs["south_hub_res_T"] = piv_res[HUB_SOUTH].values if HUB_SOUTH in piv_res else 0.0
        hubs["north_hub_res_T"] = piv_res[HUB_NORTH].values if HUB_NORTH in piv_res else 0.0
        hubs["center_hub_res_T"] = piv_res[HUB_CENTER].values if HUB_CENTER in piv_res else 0.0

        # Проверяем непрерывность 15-минутного шага для хабов (исключаем ночные скачки)
        prev_hub_dt = hubs["datetime"].shift(1)
        is_hub_c15 = (hubs["datetime"] - prev_hub_dt) == pd.Timedelta(minutes=15)
        hubs["south_hub_res_T_15"] = np.where(is_hub_c15, hubs["south_hub_res_T"].shift(1), 0.0)
        hubs["north_hub_res_T_15"] = np.where(is_hub_c15, hubs["north_hub_res_T"].shift(1), 0.0)
        hubs["center_hub_res_T_15"] = np.where(is_hub_c15, hubs["center_hub_res_T"].shift(1), 0.0)

        df = df.merge(hubs.drop(columns=["datetime"]), left_on="datetime", right_index=True, how="left").sort_values(["station_code", "datetime"]).reset_index(drop=True)
        g = df.groupby(["station_code", "month"])

        # 4. Лаги на момент T с проверкой временной непрерывности (исключаем перетекание через ночной перерыв)
        prev_dt_1 = g["datetime"].shift(1)
        prev_dt_2 = g["datetime"].shift(2)
        prev_dt_3 = g["datetime"].shift(3)

        is_c15 = (df["datetime"] - prev_dt_1) == pd.Timedelta(minutes=15)
        is_c30 = (df["datetime"] - prev_dt_2) == pd.Timedelta(minutes=30)
        is_c45 = (df["datetime"] - prev_dt_3) == pd.Timedelta(minutes=45)

        df["res_T"] = df["residual"]
        df["res_T_15"] = np.where(is_c15, g["residual"].shift(1), 0.0)
        df["res_T_30"] = np.where(is_c30, g["residual"].shift(2), 0.0)
        df["res_T_45"] = np.where(is_c45, g["residual"].shift(3), 0.0)

        df["pax_T"] = df["passengers"]
        df["pax_T_15"] = np.where(is_c15, g["passengers"].shift(1), df["base_pax"])
        df["pax_T_30"] = np.where(is_c30, g["passengers"].shift(2), df["base_pax"])

        df["res_mean_1h"] = (df["res_T"] + df["res_T_15"] + df["res_T_30"] + df["res_T_45"]) / 4.0
        lag_res_matrix = np.column_stack([df["res_T"].values, df["res_T_15"].values, df["res_T_30"].values, df["res_T_45"].values])
        df["res_std_1h"] = np.std(lag_res_matrix, axis=1)
        df["res_trend"] = df["res_T"] - df["res_T_30"]
        df["pax_trend"] = df["pax_T"] - df["pax_T_30"]
        df["pax_ratio_base_T"] = df["pax_T"] / np.maximum(df["base_pax"], 10.0)

        # 5. Сдвиг на горизонт H: k = horizon_min // 15
        k = max(1, horizon_min // 15)
        df["target_datetime"] = g["datetime"].shift(-k)
        df["target_pax"] = g["passengers"].shift(-k)
        df["base_pax_target"] = g["base_pax"].shift(-k)
        df["base_mad_target"] = g["base_mad"].shift(-k)
        df["target_res"] = g["residual"].shift(-k)
        df["target_slot_index"] = g["slot_index"].shift(-k)
        
        # Целевой календарный контекст (строго t = T + H для полной симметрии с онлайн-инференсом)
        target_day_types = self.baseline_model.assign_day_type(df["target_datetime"])
        df["day_type_code"] = target_day_types.map(DAY_TYPE_TO_CODE).fillna(0).astype(int)
        df["dow"] = df["target_datetime"].dt.dayofweek
        df["is_weekend"] = (df["dow"] >= 5).astype(int)
        df["is_friday"] = (df["dow"] == 4).astype(int)
        df["is_holiday"] = (target_day_types == "holiday").astype(int)
        df["is_pre_holiday"] = (target_day_types == "pre_holiday").astype(int)

        # Целевые циклы
        df["target_time_sin"] = np.sin(2 * np.pi * df["target_slot_index"] / 96.0)
        df["target_time_cos"] = np.cos(2 * np.pi * df["target_slot_index"] / 96.0)
        df["target_dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
        df["target_dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)

        # Отсекаем невалидные целевые точки (выход за пределы месяца/ночные разрывы)
        valid = df.dropna(subset=["target_pax", "base_pax_target", "target_slot_index", "target_datetime", "dow"]).copy()
        valid = valid[valid["target_datetime"] - valid["datetime"] == pd.Timedelta(minutes=horizon_min)].copy()
        valid["target_slot_index"] = valid["target_slot_index"].astype(int)
        
        return valid, FEATURE_COLUMNS

    def build_inference_features(
        self,
        current_time: pd.Timestamp,
        history_pax_df: pd.DataFrame,
        horizon_min: int
    ) -> pd.DataFrame:
        """
        Строит строку признаков для всех 19 станций на момент current_time
        для прогнозирования на горизонт horizon_min.
        Используется в MetroFlowPredictor.
        """
        current_time = pd.to_datetime(current_time)
        target_time = current_time + pd.Timedelta(minutes=horizon_min)
        
        # Базовый контекст для целевого момента времени
        target_day_type = self.baseline_model.assign_day_type(pd.Series([target_time])).iloc[0]
        target_slot = self.baseline_model.get_slot_index(pd.Series([target_time])).iloc[0]
        target_dow = target_time.weekday()
        
        is_weekend = int(target_dow >= 5)
        is_friday = int(target_dow == 4)
        is_holiday = int(target_day_type == "holiday")
        is_pre_holiday = int(target_day_type == "pre_holiday")
        day_type_code = DAY_TYPE_TO_CODE.get(target_day_type, 0)
        
        target_time_sin = np.sin(2 * np.pi * target_slot / 96.0)
        target_time_cos = np.cos(2 * np.pi * target_slot / 96.0)
        target_dow_sin = np.sin(2 * np.pi * target_dow / 7.0)
        target_dow_cos = np.cos(2 * np.pi * target_dow / 7.0)

        # Вычисляем базовый профиль для всех 19 станций на target_time
        all_codes = sorted(list(STATION_METADATA.keys()))
        target_df = pd.DataFrame({
            "station_code": all_codes,
            "datetime": [target_time] * len(all_codes),
            "day_type": [target_day_type] * len(all_codes),
            "slot_index": [target_slot] * len(all_codes)
        })
        target_base_df = self.baseline_model.predict(target_df)
        base_map = dict(zip(target_base_df["station_code"], target_base_df["base_pax"]))
        mad_map = dict(zip(target_base_df["station_code"], target_base_df["base_mad"]))

        # Обработка истории до current_time: автоматическая агрегация вестибюлей и дедупликация
        if history_pax_df is not None and len(history_pax_df) > 0:
            hist = history_pax_df.copy()
            hist["datetime"] = pd.to_datetime(hist["datetime"])
            hist = hist[hist["datetime"] <= current_time]
            if "station_code" in hist.columns and len(hist) > 0:
                # Если поданы данные на уровне вестибюлей (24 шт.) — агрегируем в 19 станций
                hist = hist.groupby(["datetime", "station_code"], as_index=False)["passengers"].sum()
        else:
            hist = pd.DataFrame(columns=["datetime", "station_code", "passengers"])

        # Получаем базовый профиль истории
        if len(hist) > 0:
            hist_with_base = self.baseline_model.predict(hist)
            hist_with_base["residual"] = hist_with_base["passengers"] - hist_with_base["base_pax"]
        else:
            hist_with_base = pd.DataFrame(columns=["datetime", "station_code", "passengers", "base_pax", "residual"])

        # Извлекаем состояние хабов на current_time и T - 15 мин с контролем свежести (не старше 35 мин)
        def get_hub_res(hub_code: int, dt_point: pd.Timestamp) -> float:
            if len(hist_with_base) == 0:
                return 0.0
            sub = hist_with_base[(hist_with_base["station_code"] == hub_code) & (hist_with_base["datetime"] == dt_point)]
            if len(sub) > 0:
                return float(sub["residual"].iloc[-1])
            # Если точной отметки нет, ищем ближайшую предшествующую в пределах 35 минут
            sub_prev = hist_with_base[(hist_with_base["station_code"] == hub_code) & (hist_with_base["datetime"] <= dt_point)]
            if len(sub_prev) > 0:
                last_dt = sub_prev["datetime"].iloc[-1]
                age_min = (dt_point - last_dt).total_seconds() / 60.0
                if age_min <= 35.0:
                    decay = max(0.0, 1.0 - (age_min / 60.0))
                    return float(sub_prev["residual"].iloc[-1]) * decay
            return 0.0

        south_hub_T = get_hub_res(HUB_SOUTH, current_time)
        north_hub_T = get_hub_res(HUB_NORTH, current_time)
        center_hub_T = get_hub_res(HUB_CENTER, current_time)

        south_hub_T_15 = get_hub_res(HUB_SOUTH, current_time - pd.Timedelta(minutes=15))
        north_hub_T_15 = get_hub_res(HUB_NORTH, current_time - pd.Timedelta(minutes=15))
        center_hub_T_15 = get_hub_res(HUB_CENTER, current_time - pd.Timedelta(minutes=15))

        # Временные точки для точных лагов
        t_0 = current_time
        t_15 = current_time - pd.Timedelta(minutes=15)
        t_30 = current_time - pd.Timedelta(minutes=30)
        t_45 = current_time - pd.Timedelta(minutes=45)

        rows = []
        for code in all_codes:
            meta = STATION_METADATA[code]
            base_target = base_map.get(code, 100.0)
            mad_target = mad_map.get(code, 15.0)

            st_hist = hist_with_base[hist_with_base["station_code"] == code] if len(hist_with_base) > 0 else pd.DataFrame()
            
            pax_T = base_target
            res_T = 0.0
            pax_T_15 = base_target
            res_T_15 = 0.0
            pax_T_30 = base_target
            res_T_30 = 0.0
            res_T_45 = 0.0

            if len(st_hist) > 0:
                st_by_dt = st_hist.set_index("datetime")
                
                # Извлечение t_0 (T) с учетом задержки поступления телеметрии
                if t_0 in st_by_dt.index:
                    pax_T = float(st_by_dt.loc[t_0, "passengers"])
                    res_T = float(st_by_dt.loc[t_0, "residual"])
                else:
                    # Если телеметрия на момент T задерживается, проверяем свежесть последней записи
                    latest_dt = st_by_dt.index.max()
                    age_min = (current_time - latest_dt).total_seconds() / 60.0
                    if age_min <= 20.0: # задержка 1 квант (15 мин)
                        res_T = float(st_by_dt.loc[latest_dt, "residual"]) * 0.9
                        pax_T = float(np.maximum(0.0, base_target + res_T))
                    elif age_min <= 35.0: # задержка 2 кванта (30 мин)
                        res_T = float(st_by_dt.loc[latest_dt, "residual"]) * 0.7
                        pax_T = float(np.maximum(0.0, base_target + res_T))
                    # Если данные старше 35 мин — они устарели (stale), fallback на базовый профиль

                if t_15 in st_by_dt.index:
                    pax_T_15 = float(st_by_dt.loc[t_15, "passengers"])
                    res_T_15 = float(st_by_dt.loc[t_15, "residual"])
                else:
                    pax_T_15 = pax_T
                    res_T_15 = res_T

                if t_30 in st_by_dt.index:
                    pax_T_30 = float(st_by_dt.loc[t_30, "passengers"])
                    res_T_30 = float(st_by_dt.loc[t_30, "residual"])
                else:
                    pax_T_30 = pax_T_15
                    res_T_30 = res_T_15

                if t_45 in st_by_dt.index:
                    res_T_45 = float(st_by_dt.loc[t_45, "residual"])
                else:
                    res_T_45 = res_T_30

            last4_res = [res_T, res_T_15, res_T_30, res_T_45]
            res_mean_1h = float(np.mean(last4_res))
            res_std_1h = float(np.std(last4_res))
            res_trend = float(res_T - res_T_30)
            pax_trend = float(pax_T - pax_T_30)
            pax_ratio_base_T = float(pax_T / max(base_target, 10.0))

            # Динамический поиск лага того же типа дня (res_same_daytype_last)
            res_same_daytype = 0.0
            if len(st_hist) > 0 and "day_type" in st_hist.columns and "slot_index" in st_hist.columns:
                cur_day_start = current_time.normalize()
                prior_daytype_rows = st_hist[
                    (st_hist["day_type"] == target_day_type) &
                    (st_hist["slot_index"] == target_slot) &
                    (st_hist["datetime"] < cur_day_start)
                ]
                if len(prior_daytype_rows) > 0:
                    res_same_daytype = float(prior_daytype_rows.sort_values("datetime")["residual"].iloc[-1])

            rows.append({
                "station_code": code,
                "station_name": meta["station_name"],
                "station_order": meta["station_order"],
                "has_turnaround": meta["has_turnaround"],
                "target_slot_index": target_slot,
                "day_type_code": day_type_code,
                "dow": target_dow,
                "is_weekend": is_weekend,
                "is_friday": is_friday,
                "is_holiday": is_holiday,
                "is_pre_holiday": is_pre_holiday,
                "target_time_sin": target_time_sin,
                "target_time_cos": target_time_cos,
                "target_dow_sin": target_dow_sin,
                "target_dow_cos": target_dow_cos,
                "base_pax_target": base_target,
                "base_mad_target": mad_target,
                "res_T": res_T,
                "res_T_15": res_T_15,
                "res_T_30": res_T_30,
                "res_T_45": res_T_45,
                "pax_T": pax_T,
                "pax_T_15": pax_T_15,
                "pax_T_30": pax_T_30,
                "res_mean_1h": res_mean_1h,
                "res_std_1h": res_std_1h,
                "res_trend": res_trend,
                "pax_trend": pax_trend,
                "pax_ratio_base_T": pax_ratio_base_T,
                "res_same_daytype_last": res_same_daytype,
                "south_hub_res_T": south_hub_T,
                "north_hub_res_T": north_hub_T,
                "center_hub_res_T": center_hub_T,
                "south_hub_res_T_15": south_hub_T_15,
                "north_hub_res_T_15": north_hub_T_15,
                "center_hub_res_T_15": center_hub_T_15,
                "horizon_min": horizon_min,
                "target_datetime": target_time
            })
            
        return pd.DataFrame(rows)
