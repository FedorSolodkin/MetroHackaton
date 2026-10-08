"""
Движок воспроизведения телеметрии реального времени (Real-Time Telemetry Stream Replayer).
Имитирует работу SCADA-системы Линии 1 Петербургского метрополитена.

Функционал:
- Пошаговое воспроизведение 15-минутных интервалов и интерполяция движения поездов.
- Интеграция прогнозов многогоризонтных ML-моделей (15, 30, 60, 120 мин).
- Расчет плотности платформ по стандарту Дж. Фруина (LOS A-F).
- Динамический вызов Decision Engine для выдачи директив ЦУП.
- Поддержка What-If сценариев (погодный слайдер, инциденты на перегонах).
"""

from typing import Dict, Any, List, Optional
import os
import time
import math
import datetime
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.stations import STATIONS_LINE_1, INTERSTATION_SECTIONS, get_interstation_travel_time
from configs.metro_config import METRO_CONFIG, ROLLING_STOCK

LINE_1_TOTAL_LENGTH_KM = 29.6
LINE_1_ONE_WAY_TIME_MIN = 49.5
STATIONS_BY_CODE = {s["code"]: s for s in STATIONS_LINE_1}
from src.dispatch.decision_engine import (
    DispatchParams, SlotContext, DispatchDecision, decide_line_dispatch
)
from src.utils.metrics import calculate_platform_density, calculate_train_saturation


class MetroStreamReplayer:
    """
    Потоковый симулятор телеметрии Линии 1.
    """

    def __init__(self, data_root: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = data_root or os.path.join(base_dir, "data")
        
        # 1. Загрузка предсказаний моделей и факта
        ml_pred_path = os.path.join(self.data_dir, "ml_multi_horizon_predictions.parquet")
        if os.path.exists(ml_pred_path):
            self.df_telemetry = pd.read_parquet(ml_pred_path)
        else:
            base_pred_path = os.path.join(self.data_dir, "baseline_predictions.parquet")
            self.df_telemetry = pd.read_parquet(base_pred_path)

        self.df_telemetry["datetime"] = pd.to_datetime(self.df_telemetry["datetime"])
        self.unique_timestamps = sorted(self.df_telemetry["datetime"].unique())

        # Выбираем характерный тестовый день (Среда 02.09.2026 — первый рабочий день осени)
        self.default_date = pd.to_datetime("2026-09-02").date()
        self.day_timestamps = [ts for ts in self.unique_timestamps if ts.date() == self.default_date]
        if not self.day_timestamps:
            self.day_timestamps = self.unique_timestamps[:96]

        self.current_idx = 28 # Начинаем в 07:00 (утренний пик)
        self.is_playing = False
        self.playback_speed = 5.0 # 5x по умолчанию
        self.weather_shock_mm = 0.0 # Слайдер ливня (мм/ч)
        self.active_incident: Optional[str] = None
        self.dispatch_params = DispatchParams()

        # Кумулятивный экономический счетчик за смену
        self.shift_savings_rub = 0.0

        # Кэш истории для MAD
        self.history_cache = self._build_history_cache()

    def _build_history_cache(self) -> Dict[int, Dict[int, List[float]]]:
        """Строит компактную историю притока по (station_code, interval_96) для робастной статистики."""
        cache: Dict[int, Dict[int, List[float]]] = {}
        for sc in self.df_telemetry["station_code"].unique():
            cache[int(sc)] = {}
            st_sub = self.df_telemetry[self.df_telemetry["station_code"] == sc]
            for _, row in st_sub.iterrows():
                dt = row["datetime"]
                slot_idx = dt.hour * 4 + (dt.minute // 15)
                val = float(row.get("actual_15m", row.get("passengers", 0.0)))
                if slot_idx not in cache[int(sc)]:
                    cache[int(sc)][slot_idx] = []
                cache[int(sc)][slot_idx].append(val)
        return cache

    def set_simulation_date(self, target_date_str: str) -> None:
        """Переключение даты симуляции."""
        target_date = pd.to_datetime(target_date_str).date()
        matched = [ts for ts in self.unique_timestamps if ts.date() == target_date]
        if matched:
            self.day_timestamps = matched
            self.current_idx = min(28, len(matched) - 1)

    def set_time_index(self, idx: int) -> None:
        """Перемотка на конкретный временной интервал (0..95)."""
        self.current_idx = max(0, min(idx, len(self.day_timestamps) - 1))

    def step(self) -> Dict[str, Any]:
        """Переход на следующий 15-минутный шаг симуляции."""
        if self.current_idx < len(self.day_timestamps) - 1:
            self.current_idx += 1
        else:
            self.current_idx = 0 # Зацикливание суточного цикла
        return self.get_current_state()

    def get_current_state(self) -> Dict[str, Any]:
        """
        Формирует комплексный снимок состояния Линии 1 для SCADA Cockpit.
        """
        cur_dt = self.day_timestamps[self.current_idx]
        slot_str = cur_dt.strftime("%H:%M")
        interval_96 = cur_dt.hour * 4 + (cur_dt.minute // 15)

        # Выбираем срез телеметрии на текущий момент
        cur_slice = self.df_telemetry[self.df_telemetry["datetime"] == cur_dt].sort_values("station_order")

        # Применяем погодный слайдер What-If
        weather_factor = 1.0 + (self.weather_shock_mm * 0.035) # +3.5% притока на каждый мм/ч дождя

        # 1. Станционная телеметрия
        stations_state = []
        total_inflow = 0.0
        center_inflow = 0.0

        for _, row in cur_slice.iterrows():
            sc = int(row["station_code"])
            st_info = STATIONS_BY_CODE.get(sc, {})
            base_flow = float(row.get("actual_15m", row.get("passengers", 0.0)))
            actual_flow = round(base_flow * weather_factor, 1)

            pred_15 = round(float(row.get("pred_ml_15m", base_flow * 1.02)) * weather_factor, 1)
            pred_30 = round(float(row.get("pred_ml_30m", base_flow * 1.05)) * weather_factor, 1)
            pred_60 = round(float(row.get("pred_ml_60m", base_flow * 0.98)) * weather_factor, 1)
            pred_120 = round(float(row.get("pred_ml_120m", base_flow * 0.95)) * weather_factor, 1)

            total_inflow += actual_flow
            if sc in [117, 118, 119, 120, 121, 122]: # Центральный участок («Балтийская» – «Пл. Ленина»)
                center_inflow += actual_flow

            # Плотность платформы по Фруину
            platform_area = float(st_info.get("platform_area_m2", 700.0))
            density_info = calculate_platform_density(
                inflow_15min=actual_flow,
                dwell_fraction=0.20,
                platform_area_m2=platform_area
            )

            stations_state.append({
                "code": sc,
                "name": st_info.get("name_ru", row.get("station_name", "")),
                "order": int(st_info.get("order", row.get("station_order", 0))),
                "type": st_info.get("type", "standard"),
                "has_turnaround": bool(st_info.get("has_turnaround", False)),
                "inflow_15m": actual_flow,
                "pred_15m": pred_15,
                "pred_30m": pred_30,
                "pred_60m": pred_60,
                "pred_120m": pred_120,
                "density_p_m2": round(float(density_info["density_pers_m2"]), 2),
                "fruin_los": str(density_info["los_level"]),
                "is_critical": bool(density_info["is_los_danger"])
            })

        # 2. Определение текущей парности и поездов на линии
        # График движения для рабочего дня сентября:
        hour = cur_dt.hour
        if 7 <= hour < 10:
            planned_trains = 50
            planned_pairs = 30
        elif 10 <= hour < 16:
            planned_trains = 33
            planned_pairs = 20
        elif 16 <= hour < 19:
            planned_trains = 50
            planned_pairs = 30
        elif 19 <= hour < 22:
            planned_trains = 25
            planned_pairs = 15
        elif 22 <= hour or hour < 6:
            planned_trains = 15
            planned_pairs = 9
        else:
            planned_trains = 30
            planned_pairs = 18

        # Насыщенность линии
        sat_info = calculate_train_saturation(
            passengers_15min=total_inflow,
            trains_on_line=planned_trains,
            is_line_wide_inflow=True
        )

        # 3. Интеллектуальный расчет решения диспетчера (Decision Engine)
        future_slots = []
        for offset in range(1, 9):
            fut_idx = min(self.current_idx + offset, len(self.day_timestamps) - 1)
            fut_dt = self.day_timestamps[fut_idx]
            fut_time = fut_dt.strftime("%H:%M")
            fut_slot_idx = fut_dt.hour * 4 + (fut_dt.minute // 15)

            # Прогноз суммарного притока
            fut_slice = self.df_telemetry[self.df_telemetry["datetime"] == fut_dt]
            col_pred = f"pred_ml_{offset*15}m" if f"pred_ml_{offset*15}m" in fut_slice.columns else "pred_ml_15m"
            fut_sum = float(fut_slice[col_pred].sum()) * weather_factor if col_pred in fut_slice.columns else total_inflow * 1.05

            # Исторические значения слота
            hist_vals = []
            for sc, slot_dict in self.history_cache.items():
                if fut_slot_idx in slot_dict:
                    hist_vals.extend(slot_dict[fut_slot_idx][:3])

            future_slots.append(SlotContext(
                time=fut_time,
                forecast=fut_sum,
                history=hist_vals or [fut_sum * 0.95],
                trains_now=planned_pairs,
                is_central_concentrated=(center_inflow / max(total_inflow, 1.0) > 0.42),
                primary_depot="severnoe" if (center_inflow / max(total_inflow, 1.0) > 0.42) else "avtovo"
            ))

        decision = decide_line_dispatch(future_slots, self.dispatch_params)

        # Если активно погодное возмущение, добавляем объяснение
        dispatcher_reasons = []
        if self.weather_shock_mm > 0:
            dispatcher_reasons.append(f"Погодное возмущение: ливень {self.weather_shock_mm:.1f} мм/ч (+{self.weather_shock_mm * 3.5:.1f}% к потоку)")
        if decision.action == "short_turn":
            dispatcher_reasons.append("Концентрация перегруза в центральном тоннеле (Чернышевская - Пл. Восстания)")
            dispatcher_reasons.append("Зонный оборот по ст. 'Площадь Ленина' без холостого прогона до 'Девяткино'")
        elif decision.action == "release_hot":
            dispatcher_reasons.append("Прогнозируется выход коэффициента насыщенности за 1.00 через 30 минут")
            dispatcher_reasons.append(f"Выдача горячего резерва из {decision.location}")
        elif decision.action == "return":
            dispatcher_reasons.append("Спад пассажиропотока после окончания пика: экономия тяговой электроэнергии")

        # Накопление экономического эффекта
        if decision.action in ["short_turn", "return"]:
            self.shift_savings_rub += (abs(decision.economic_impact_rub) / 100.0)

        # 4. Моделирование положения поездов на линии для визуализации SCADA
        trains_positions = self._generate_train_positions(planned_trains, cur_dt)

        return {
            "timestamp": cur_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "time_slot": slot_str,
            "interval_index": interval_96,
            "weather": {
                "temperature": 15.8,
                "rain_mm": self.weather_shock_mm,
                "is_rain": self.weather_shock_mm > 0.1,
                "condition": "Сильный ливень" if self.weather_shock_mm > 5.0 else ("Дождь" if self.weather_shock_mm > 0.5 else "Ясно")
            },
            "line_summary": {
                "total_inflow_15m": round(total_inflow),
                "trains_on_line": planned_trains,
                "planned_pairs": planned_pairs,
                "capacity_15min": round(float(sat_info["capacity_15min"])),
                "saturation_ratio": round(float(sat_info["saturation_ratio"]), 3),
                "is_overcrowded": bool(sat_info["is_overcrowded"]),
                "is_critical": bool(sat_info["is_critical"]),
                "cumulative_savings_rub": round(self.shift_savings_rub)
            },
            "stations": stations_state,
            "decision": {
                "action": decision.action,
                "count": decision.count,
                "start_time": decision.start_time,
                "end_time": decision.end_time,
                "location": decision.location,
                "z_score": decision.z_score,
                "economic_impact_rub": decision.economic_impact_rub,
                "reason": decision.reason,
                "dispatcher_reasons": dispatcher_reasons,
                "timer_seconds": 900 - ((cur_dt.minute % 15) * 60 + cur_dt.second)
            },
            "trains": trains_positions
        }

    def _generate_train_positions(self, num_trains: int, cur_dt: datetime.datetime) -> List[Dict[str, Any]]:
        """
        Генерирует физические координаты поездов вдоль двух путей Линии 1.
        Track 1: Пр. Ветеранов (0 км) -> Девяткино (29.6 км)
        Track 2: Девяткино (29.6 км) -> Пр. Ветеранов (0 км)
        """
        trains = []
        total_time_min = 99.0 # Полный круг
        minute_of_day = cur_dt.hour * 60 + cur_dt.minute + (cur_dt.second / 60.0)

        for i in range(num_trains):
            # Фазовый сдвиг каждого поезда
            train_phase = (minute_of_day + (i * (total_time_min / max(num_trains, 1)))) % total_time_min
            
            if train_phase < LINE_1_ONE_WAY_TIME_MIN:
                # Движение на север (Track 1: Ветеранов -> Девяткино)
                direction = "north"
                track = 1
                progress = train_phase / LINE_1_ONE_WAY_TIME_MIN
                km = progress * LINE_1_TOTAL_LENGTH_KM
            else:
                # Движение на юг (Track 2: Девяткино -> Ветеранов)
                direction = "south"
                track = 2
                progress = (train_phase - LINE_1_ONE_WAY_TIME_MIN) / LINE_1_ONE_WAY_TIME_MIN
                km = LINE_1_TOTAL_LENGTH_KM - (progress * LINE_1_TOTAL_LENGTH_KM)

            trains.append({
                "train_id": 101 + i,
                "track": track,
                "direction": direction,
                "progress_pct": round(progress * 100, 1),
                "km": round(km, 2),
                "model": "81-725/726/727 «Балтиец»" if i % 2 == 0 else "81-722/723/724 «Юбилейный»"
            })

        return trains
