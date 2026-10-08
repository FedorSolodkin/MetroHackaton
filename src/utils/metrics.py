"""
Метрики качества моделей и транспортно-экономические расчеты для Линии 1.
Содержит расчет WAPE, насыщения линии поездами, плотности пассажиров на платформах
и финансового калькулятора тяговой электроэнергии и ТОиР.
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error (WAPE).
    Отраслевой транспортный стандарт оценки прогноза пассажиропотока:
    WAPE = sum(|y - y_hat|) / sum(y)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    total_actual = np.sum(y_true)
    if total_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (пассажиров в интервал)."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Коэффициент детерминации R^2."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - (ss_res / ss_tot))


def calculate_train_saturation(
    passengers_15min: Union[float, np.ndarray],
    trains_on_line: Union[int, np.ndarray],
    turnaround_min: float = 99.0,
    comfort_capacity: int = 1478,
    is_line_wide_inflow: bool = True,
    turnover_ratio: float = 3.0,
    directional_split: float = 0.65
) -> Dict[str, Any]:
    """
    Расчет коэффициента насыщенности линии подвижным составом.
    
    turnaround_min: время полного оборота (99 минут).
    В 15 минут через средний перегон проходит: (trains_on_line / turnaround_min) * 15 поездов в каждую сторону.
    Провозная способность за 15 минут в одном направлении:
    capacity_15min = (trains_on_line / turnaround_min) * 15 * comfort_capacity
    
    Если passengers_15min представляет суммарный приток по всем 19 станциям линии (is_line_wide_inflow=True),
    нагрузка на наиболее загруженный лимитирующий перегон в пиковом направлении оценивается с учетом
    коэффициента сменяемости (turnover_ratio ~ 3.0) и направленности потока (directional_split ~ 0.65).
    """
    t_on_line = np.asarray(trains_on_line, dtype=float)
    p_inflow = np.asarray(passengers_15min, dtype=float)
    
    trains_passing_15min = (t_on_line / turnaround_min) * 15.0
    capacity_15min = trains_passing_15min * comfort_capacity
    
    if is_line_wide_inflow:
        # Оценка пассажиров на наиболее загруженном перегоне пикового направления
        peak_section_load = (p_inflow * directional_split) / turnover_ratio
    else:
        peak_section_load = p_inflow
        
    # Предотвращаем деление на околонулевую емкость в ночные часы закрытия метро
    valid_mask = capacity_15min > 100.0
    saturation_ratio = np.where(
        valid_mask,
        peak_section_load / np.maximum(capacity_15min, 1.0),
        0.0
    )
    
    # Также сохраняем прямое отношение суммарного входа к однопутной емкости для визуального сопоставления
    raw_ratio = np.where(valid_mask, p_inflow / np.maximum(capacity_15min, 1.0), 0.0)
    
    return {
        "trains_passing_15min": trains_passing_15min,
        "capacity_15min": capacity_15min,
        "peak_section_load": peak_section_load,
        "saturation_ratio": saturation_ratio,
        "raw_saturation_ratio": raw_ratio,
        "is_overcrowded": saturation_ratio > 1.0,
        "is_critical": saturation_ratio > 1.15,
        "is_underloaded": (saturation_ratio < 0.65) & valid_mask
    }


def calculate_platform_density(
    inflow_15min: Union[float, np.ndarray],
    dwell_fraction: float = 0.20,
    platform_area_m2: float = 650.0
) -> Dict[str, Any]:
    """
    Оценка плотности скопления пассажиров на платформе станции (чел/м²).
    Стандартная платформа СПб Линии 1: полезная зона ожидания перед дверьми ~600–750 м².
    dwell_fraction: доля входящих пассажиров, накапливающихся на платформе между прибытиями поездов.
    
    Уровни обслуживания по шкале Дж. Фруина (Fruin Level of Service - LOS):
    - LOS A-B: < 1.11 чел/м² (свободная циркуляция) -> GREEN
    - LOS C-D: 1.11 - 3.33 чел/м² (затрудненное движение, плотный поток) -> YELLOW
    - LOS E-F: > 3.33 чел/м² (критическая давка, риск падения на пути) -> RED
    """
    passengers_on_platform = np.asarray(inflow_15min, dtype=float) * dwell_fraction
    density = passengers_on_platform / platform_area_m2
    
    los_level = np.where(
        density > 3.33,
        "RED",
        np.where(density > 1.11, "YELLOW", "GREEN")
    )
    
    return {
        "density_pers_m2": density,
        "los_level": los_level,
        "is_los_danger": density > 3.33
    }


def calculate_economic_effect(
    trips_optimized: int,
    is_full_circle: bool = True,
    kwh_cost_rub: float = 6.8,
    train_type: str = "baltiec"
) -> Dict[str, float]:
    """
    Экономический калькулятор оптимизации насыщенности линии.
    Снятие избыточного рейса в межпиковый спад экономит тяговую электроэнергию
    и ресурс до регламентного ТО-2 / ПДР.
    
    L_full: 59.2 км (полный оборот)
    L_short: 28.4 км (зонный оборот ст. Автово - ст. Пл. Ленина)
    
    Физический норматив расхода электроэнергии 8-вагонного поезда метро:
    ~24.5 кВт⋅ч на поезд-км (~3.06 кВт⋅ч на вагоно-км, ~1.45 МВт⋅ч на полный оборот 59.2 км).
    """
    trip_km = 59.2 if is_full_circle else 28.4
    kwh_per_train_km = 24.5  # Физически выверенный удельный расход 8-вагонного состава
    
    total_km_saved = trips_optimized * trip_km
    total_kwh_saved = total_km_saved * kwh_per_train_km
    energy_rub_saved = total_kwh_saved * kwh_cost_rub
    
    # Стоимость амортизации пробега ТОиР (~120 руб на поезд-км)
    toir_rub_saved = total_km_saved * 120.0
    total_rub_saved = energy_rub_saved + toir_rub_saved
    
    return {
        "trips_optimized": float(trips_optimized),
        "total_km_saved": float(total_km_saved),
        "total_kwh_saved": float(total_kwh_saved),
        "energy_rub_saved": float(energy_rub_saved),
        "toir_rub_saved": float(toir_rub_saved),
        "total_rub_saved": float(total_rub_saved)
    }

calculate_energy_and_toir_savings = calculate_economic_effect
