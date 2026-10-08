"""
Исторический медианный бейзлайн (Historical Median Baseline) для Линии 1 Петербургского метрополитена.
Официальная точка отсчета (Benchmark) транспортных прогнозов.

Принципы валидации:
- Строго хронологическое разделение: Обучение на феврале, мае, июле 2026 (90 дней).
- Тестирование на отложенной выборке сентября 2026 (30 дней).
- Полное отсутствие Data Leakage: медианы рассчитываются строго по обучающей выборке.
- Иерархический fallback для редких дней (праздники / предпраздники).
"""

from typing import Dict, Any, Optional, Tuple
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.data_loader import MetroDataLoader
from src.data.calendar_features import CalendarFeatureEngine
from src.utils.metrics import calculate_wape, calculate_mae, calculate_rmse, calculate_r2
from configs.stations import STATIONS_LINE_1


class HistoricalMedianBaseline:
    """
    Бейзлайн на основе исторических медианных профилей пассажиропотока.
    Использует иерархическую агрегацию:
    1. Точный срез: (station_code, day_of_week, interval_96)
    2. Тип дня: (station_code, day_type, interval_96) для праздников/предпраздников
    3. Выходной/будний: (station_code, is_weekend, interval_96)
    4. Общий суточный профиль станции: (station_code, interval_96)
    """

    def __init__(self):
        self.median_dow_: Optional[pd.DataFrame] = None
        self.median_day_type_: Optional[pd.DataFrame] = None
        self.median_weekend_: Optional[pd.DataFrame] = None
        self.median_interval_: Optional[pd.DataFrame] = None
        self.global_median_: float = 0.0
        self.calendar_engine = CalendarFeatureEngine()

    def fit(self, train_df: pd.DataFrame, target_col: str = "passengers") -> "HistoricalMedianBaseline":
        """
        Обучение бейзлайна: расчет медианных профилей строго по обучающей выборке.
        """
        df = train_df.copy()
        if "interval_96" not in df.columns or "day_of_week" not in df.columns:
            df = self.calendar_engine.add_features(df, datetime_col="datetime")

        # 1. Точный срез по дню недели и 15-мин интервалу
        self.median_dow_ = (
            df.groupby(["station_code", "day_of_week", "interval_96"])[target_col]
            .median()
            .reset_index()
            .rename(columns={target_col: "pred_dow"})
        )

        # 2. Срез по типу дня (workday, weekend, holiday, pre_holiday)
        self.median_day_type_ = (
            df.groupby(["station_code", "day_type", "interval_96"])[target_col]
            .median()
            .reset_index()
            .rename(columns={target_col: "pred_day_type"})
        )

        # 3. Срез по признаку выходного дня
        self.median_weekend_ = (
            df.groupby(["station_code", "is_weekend", "interval_96"])[target_col]
            .median()
            .reset_index()
            .rename(columns={target_col: "pred_weekend"})
        )

        # 4. Общий суточный профиль станции
        self.median_interval_ = (
            df.groupby(["station_code", "interval_96"])[target_col]
            .median()
            .reset_index()
            .rename(columns={target_col: "pred_station_interval"})
        )

        self.global_median_ = float(df[target_col].median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Генерация предсказаний с применением иерархического fallback.
        """
        if self.median_dow_ is None:
            raise ValueError("Модель не обучена. Вызовите .fit() перед .predict().")

        eval_df = df.copy()
        if "interval_96" not in eval_df.columns or "day_of_week" not in eval_df.columns:
            eval_df = self.calendar_engine.add_features(eval_df, datetime_col="datetime")

        eval_df["_row_id"] = np.arange(len(eval_df))

        # Мерджим иерархические уровни
        # 1. По типу дня (особенно важно для праздников и предпраздников)
        m_dt = pd.merge(
            eval_df[["_row_id", "station_code", "day_type", "interval_96"]],
            self.median_day_type_,
            on=["station_code", "day_type", "interval_96"],
            how="left"
        )

        # 2. По дню недели
        m_dow = pd.merge(
            eval_df[["_row_id", "station_code", "day_of_week", "interval_96"]],
            self.median_dow_,
            on=["station_code", "day_of_week", "interval_96"],
            how="left"
        )

        # 3. По выходному/буднему
        m_wk = pd.merge(
            eval_df[["_row_id", "station_code", "is_weekend", "interval_96"]],
            self.median_weekend_,
            on=["station_code", "is_weekend", "interval_96"],
            how="left"
        )

        # 4. По интервалу станции
        m_st = pd.merge(
            eval_df[["_row_id", "station_code", "interval_96"]],
            self.median_interval_,
            on=["station_code", "interval_96"],
            how="left"
        )

        # Иерархический выбор:
        # Если праздник или предпраздник -> приоритет pred_day_type
        # Иначе -> приоритет pred_dow -> pred_weekend -> pred_station_interval -> global_median
        preds = np.full(len(eval_df), np.nan)

        is_special_day = eval_df["day_type"].isin(["holiday", "pre_holiday"]).values
        dt_vals = m_dt["pred_day_type"].values
        dow_vals = m_dow["pred_dow"].values
        wk_vals = m_wk["pred_weekend"].values
        st_vals = m_st["pred_station_interval"].values

        # Для праздников берем сначала профиль праздника
        preds[is_special_day] = dt_vals[is_special_day]

        # Для остальных берем dow
        normal_mask = ~is_special_day
        preds[normal_mask] = dow_vals[normal_mask]

        # Fallback уровни
        preds = np.where(np.isnan(preds), dow_vals, preds)
        preds = np.where(np.isnan(preds), dt_vals, preds)
        preds = np.where(np.isnan(preds), wk_vals, preds)
        preds = np.where(np.isnan(preds), st_vals, preds)
        preds = np.where(np.isnan(preds), self.global_median_, preds)

        # Пассажиропоток не может быть отрицательным
        preds = np.maximum(preds, 0.0)
        return preds


def evaluate_baseline_pipeline(
    data_root: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Запуск полного цикла оценки исторического бейзлайна:
    1. Загрузка агрегированного 15-минутного пассажиропотока (19 станций Линии 1).
    2. Строгое хронологическое разбиение:
       - Train: Февраль, Май, Июль 2026 (90 дней, ~66 384 строк)
       - Test: Сентябрь 2026 (30 дней, ~22 800 строк)
    3. Обучение HistoricalMedianBaseline на Train.
    4. Предсказание на Test.
    5. Расчет метрик (WAPE, MAE, RMSE, R^2):
       - Общая по всей Линии 1
       - Попикетная (по каждой из 19 станций)
       - По периодам (Утренний пик 07-10, Вечерний пик 17-20, Межпик, Ночь)
    6. Сохранение предсказаний в parquet и графиков в png.
    """
    loader = MetroDataLoader(data_root=data_root)
    df_15m = loader.load_15min_dataset()
    df_station = loader.get_station_aggregated_flow(df_15m)

    # Добавляем календарные признаки
    cal = CalendarFeatureEngine()
    df_station = cal.add_features(df_station, datetime_col="datetime")

    # Хронологическое деление
    # Train: месяцы 2, 5, 7
    # Test: месяц 9 (сентябрь 2026)
    train_mask = df_station["datetime"].dt.month.isin([2, 5, 7])
    test_mask = df_station["datetime"].dt.month == 9

    train_df = df_station[train_mask].copy().reset_index(drop=True)
    test_df = df_station[test_mask].copy().reset_index(drop=True)

    print(f"Dataset split: Train = {len(train_df)} rows ({train_df['datetime'].min().date()} .. {train_df['datetime'].max().date()})")
    print(f"Dataset split: Test  = {len(test_df)} rows ({test_df['datetime'].min().date()} .. {test_df['datetime'].max().date()})")

    # Обучение
    baseline = HistoricalMedianBaseline()
    baseline.fit(train_df, target_col="passengers")

    # Предсказание
    y_test_true = test_df["passengers"].values
    y_test_pred = baseline.predict(test_df)
    test_df["baseline_pred"] = y_test_pred
    test_df["abs_error"] = np.abs(y_test_true - y_test_pred)

    # 1. Общие метрики на тесте (September 2026)
    wape_total = calculate_wape(y_test_true, y_test_pred)
    mae_total = calculate_mae(y_test_true, y_test_pred)
    rmse_total = calculate_rmse(y_test_true, y_test_pred)
    r2_total = calculate_r2(y_test_true, y_test_pred)

    # Метрика на суммарном пассажиропотоке всей линии (Line-wide aggregated flow)
    line_true = test_df.groupby("datetime")["passengers"].sum()
    line_pred = test_df.groupby("datetime")["baseline_pred"].sum()
    line_wape = calculate_wape(line_true.values, line_pred.values)
    line_mae = calculate_mae(line_true.values, line_pred.values)
    line_rmse = calculate_rmse(line_true.values, line_pred.values)
    line_r2 = calculate_r2(line_true.values, line_pred.values)

    # 2. Метрики по станциям
    station_metrics = []
    for sc, group in test_df.groupby("station_code"):
        s_name = group["station_name"].iloc[0]
        s_order = group["station_order"].iloc[0]
        s_true = group["passengers"].values
        s_pred = group["baseline_pred"].values
        s_wape = calculate_wape(s_true, s_pred)
        s_mae = calculate_mae(s_true, s_pred)
        s_rmse = calculate_rmse(s_true, s_pred)
        s_r2 = calculate_r2(s_true, s_pred)
        station_metrics.append({
            "station_code": int(sc),
            "station_order": int(s_order),
            "station_name": str(s_name),
            "wape": float(round(s_wape * 100, 2)),
            "mae": float(round(s_mae, 1)),
            "rmse": float(round(s_rmse, 1)),
            "r2": float(round(s_r2, 3)),
            "total_passengers": int(s_true.sum())
        })

    station_metrics_df = pd.DataFrame(station_metrics).sort_values("station_order")

    # 3. Метрики по временным периодам
    hour = test_df["hour"]
    morning_mask = (hour >= 7) & (hour < 10)
    evening_mask = (hour >= 17) & (hour < 20)
    day_mask = (hour >= 10) & (hour < 17)
    night_mask = (hour >= 20) | (hour < 7)

    period_metrics = {
        "morning_peak_07_10": {
            "wape": float(round(calculate_wape(y_test_true[morning_mask], y_test_pred[morning_mask]) * 100, 2)),
            "mae": float(round(calculate_mae(y_test_true[morning_mask], y_test_pred[morning_mask]), 1)),
        },
        "evening_peak_17_20": {
            "wape": float(round(calculate_wape(y_test_true[evening_mask], y_test_pred[evening_mask]) * 100, 2)),
            "mae": float(round(calculate_mae(y_test_true[evening_mask], y_test_pred[evening_mask]), 1)),
        },
        "daytime_offpeak_10_17": {
            "wape": float(round(calculate_wape(y_test_true[day_mask], y_test_pred[day_mask]) * 100, 2)),
            "mae": float(round(calculate_mae(y_test_true[day_mask], y_test_pred[day_mask]), 1)),
        },
        "night_offpeak": {
            "wape": float(round(calculate_wape(y_test_true[night_mask], y_test_pred[night_mask]) * 100, 2)),
            "mae": float(round(calculate_mae(y_test_true[night_mask], y_test_pred[night_mask]), 1)),
        }
    }

    # Итоговый словарь
    results = {
        "model_name": "HistoricalMedianBaseline",
        "train_period": "2026-02, 2026-05, 2026-07 (90 days)",
        "test_period": "2026-09 (30 days)",
        "station_level_metrics": {
            "wape_pct": round(wape_total * 100, 2),
            "mae": round(mae_total, 2),
            "rmse": round(rmse_total, 2),
            "r2": round(r2_total, 4)
        },
        "line_wide_metrics": {
            "wape_pct": round(line_wape * 100, 2),
            "mae": round(line_mae, 1),
            "rmse": round(line_rmse, 1),
            "r2": round(line_r2, 4)
        },
        "period_metrics": period_metrics,
        "station_metrics": station_metrics_df.to_dict(orient="records")
    }

    # Сохраняем результаты
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_dir = output_dir or os.path.join(base_dir, "data")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Сохранение предсказаний в parquet
    test_pred_path = os.path.join(save_dir, "baseline_predictions.parquet")
    test_df[["datetime", "station_code", "station_name", "station_order", "passengers", "baseline_pred", "abs_error"]].to_parquet(test_pred_path, index=False)
    print(f"Saved predictions to: {test_pred_path}")

    # 2. Сохранение метрик в json
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, "baseline_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to: {json_path}")

    # 3. Построение графика ошибок и сравнения
    figures_dir = os.path.join(base_dir, "eda", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, "baseline_performance.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#0F172A")

    for ax in axes.flat:
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors="#94A3B8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # График 1: WAPE по 19 станциям
    bars = axes[0, 0].barh(station_metrics_df["station_name"], station_metrics_df["wape"], color="#38BDF8", alpha=0.85)
    axes[0, 0].axvline(wape_total * 100, color="#F43F5E", linestyle="--", linewidth=2, label=f"Средняя WAPE: {wape_total*100:.1f}%")
    axes[0, 0].set_title("WAPE (%) Historical Baseline по станциям Линии 1 (Сентябрь 2026)", color="#F8FAFC", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("WAPE (%)", color="#94A3B8")
    axes[0, 0].legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    # График 2: Суммарный поток по Линии 1 (Первая неделя сентября: Факт vs Бейзлайн)
    one_week_line = line_true.loc["2026-09-01":"2026-09-07"]
    one_week_pred = line_pred.loc["2026-09-01":"2026-09-07"]
    axes[0, 1].plot(one_week_line.index, one_week_line.values, label="Фактический пассажиропоток", color="#38BDF8", linewidth=1.8)
    axes[0, 1].plot(one_week_pred.index, one_week_pred.values, label=f"Historical Baseline (WAPE={line_wape*100:.1f}%)", color="#F59E0B", linestyle="--", linewidth=1.5)
    axes[0, 1].set_title("Суммарный приток на Линию 1: Факт vs Бейзлайн (1-7 Сентября 2026)", color="#F8FAFC", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Пассажиров в 15 минут", color="#94A3B8")
    axes[0, 1].legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    # График 3: Суточный профиль крупного хаба (Девяткино, код 129) в рабочий день
    dev_sept = test_df[(test_df["station_code"] == 129) & (test_df["datetime"].dt.date == pd.to_datetime("2026-09-02").date())]
    axes[1, 0].plot(dev_sept["time_slot"], dev_sept["passengers"], label="Факт (Ср 02.09.2026)", color="#10B981", linewidth=2.0, marker="o", markersize=3)
    axes[1, 0].plot(dev_sept["time_slot"], dev_sept["baseline_pred"], label="Historical Baseline", color="#F59E0B", linestyle="--", linewidth=1.8)
    axes[1, 0].set_title("ст. Девяткино: Утренний пик Факт vs Бейзлайн", color="#F8FAFC", fontsize=11, fontweight="bold")
    axes[1, 0].set_xticks(range(0, 96, 8))
    axes[1, 0].set_xticklabels([dev_sept["time_slot"].iloc[i] for i in range(0, 96, 8)], rotation=45, color="#94A3B8")
    axes[1, 0].set_ylabel("Пассажиров в 15 мин", color="#94A3B8")
    axes[1, 0].legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    # График 4: Распределение остатков (Residuals = Actual - Baseline)
    residuals = y_test_true - y_test_pred
    axes[1, 1].hist(residuals, bins=60, color="#818CF8", alpha=0.8, edgecolor="#1E293B")
    axes[1, 1].axvline(0, color="#F43F5E", linestyle="--", linewidth=1.5)
    axes[1, 1].set_title(f"Распределение остатков прогноза (MAE = {mae_total:.1f}, RMSE = {rmse_total:.1f})", color="#F8FAFC", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Ошибка (Факт - Бейзлайн, пасс/15мин)", color="#94A3B8")
    axes[1, 1].set_ylabel("Частота", color="#94A3B8")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved baseline performance plot to: {fig_path}")

    return results


if __name__ == "__main__":
    print("Запуск строгой оценки Historical Median Baseline на отложенной выборке сентября 2026...")
    res = evaluate_baseline_pipeline()
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ИСТОРИЧЕСКОГО БЕЙЗЛАЙНА (SEPTEMBER 2026 TEST):")
    print(f"Station-level WAPE: {res['station_level_metrics']['wape_pct']}% (MAE: {res['station_level_metrics']['mae']}, RMSE: {res['station_level_metrics']['rmse']}, R2: {res['station_level_metrics']['r2']})")
    print(f"Line-wide WAPE:    {res['line_wide_metrics']['wape_pct']}% (MAE: {res['line_wide_metrics']['mae']}, RMSE: {res['line_wide_metrics']['rmse']}, R2: {res['line_wide_metrics']['r2']})")
    print("WAPE по периодам:")
    for period, p_res in res["period_metrics"].items():
        print(f"  {period}: WAPE = {p_res['wape']}%, MAE = {p_res['mae']}")
    print("="*60)
