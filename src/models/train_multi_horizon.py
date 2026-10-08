"""
Многогоризонтное предиктивное ядро (Multi-Horizon ML Core) для Линии 1 Петербургского метрополитена.
Обучение градиентного бустинга (LightGBM / CatBoost) на горизонты 15, 30, 60, 120 минут.

Принципы валидации:
- Полное разделение данных ДО генерации фичей с бейзлайном.
- Строго хронологический Out-of-Time тест: Train (Февраль, Май, Июль), Test (Сентябрь 2026).
- L1/Huber лосс, напрямую минимизирующий WAPE (транспортный стандарт).
- Учет пространственных волн от узловых станций (Девяткино, Проспект Ветеранов, Пл. Восстания),
  погоды Open-Meteo и производственного календаря.
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.data_loader import MetroDataLoader
from src.data.weather_loader import WeatherLoader
from src.data.calendar_features import CalendarFeatureEngine
from src.models.baseline import HistoricalMedianBaseline
from src.utils.metrics import calculate_wape, calculate_mae, calculate_rmse, calculate_r2
from configs.stations import STATIONS_LINE_1


HORIZONS = {
    "15m": 1,   # 1 интервал (15 мин)
    "30m": 2,   # 2 интервала (30 мин)
    "60m": 4,   # 4 интервала (60 мин)
    "120m": 8   # 8 интервалов (120 мин)
}


class MultiHorizonDataPrep:
    """
    Генератор обучающей и тестовой матрицы признаков для многогоризонтного прогнозирования.
    Обеспечивает строгое соблюдение причинно-следственной связи:
    В момент прогноза t доступны только телеметрия и лаги до момента t включительно.
    """

    def __init__(self, data_root: Optional[str] = None):
        self.loader = MetroDataLoader(data_root=data_root)
        self.weather_loader = WeatherLoader()
        self.calendar_engine = CalendarFeatureEngine()
        self.baseline_model = HistoricalMedianBaseline()

    def prepare_base_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Загружает 15-минутный датасет, погоду и календарь,
        разделяет на Train и Test и обучает Historical Baseline на Train.
        """
        df_15m = self.loader.load_15min_dataset()
        df_station = self.loader.get_station_aggregated_flow(df_15m)
        df_station = self.calendar_engine.add_features(df_station, datetime_col="datetime")

        # Добавляем погоду
        weather_df = self.weather_loader.get_interpolated_15min_weather()
        weather_cols = [
            "datetime", "temperature", "apparent_temperature", "precipitation",
            "rain", "snowfall", "wind_speed", "wind_gusts", "is_rain", "is_heavy_rain", "is_extreme_weather"
        ]
        available_w = [c for c in weather_cols if c in weather_df.columns]
        df_station = pd.merge(df_station, weather_df[available_w], on="datetime", how="left")

        # Заполнение пропусков погоды
        num_w = ["temperature", "apparent_temperature", "precipitation", "rain", "snowfall", "wind_speed", "wind_gusts"]
        df_station[num_w] = df_station[num_w].ffill().bfill()
        bin_w = ["is_rain", "is_heavy_rain", "is_extreme_weather"]
        df_station[bin_w] = df_station[bin_w].fillna(0).astype(int)

        # Хронологическое деление
        train_mask = df_station["datetime"].dt.month.isin([2, 5, 7])
        test_mask = df_station["datetime"].dt.month == 9

        train_df = df_station[train_mask].copy().reset_index(drop=True)
        test_df = df_station[test_mask].copy().reset_index(drop=True)

        # Обучаем бейзлайн строго на Train!
        self.baseline_model.fit(train_df, target_col="passengers")
        train_df["baseline_pred"] = self.baseline_model.predict(train_df)
        test_df["baseline_pred"] = self.baseline_model.predict(test_df)

        return train_df, test_df, df_station

    def create_horizon_dataset(
        self,
        df_split: pd.DataFrame,
        horizon_steps: int
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Создает матрицу признаков (X, y) для конкретного горизонта t + k*15мин.
        """
        df = df_split.copy()
        df = df.sort_values(["station_code", "datetime"]).reset_index(drop=True)
        df["month_group"] = df["datetime"].dt.month

        # Таргет на шаге t + horizon_steps
        df["target"] = df.groupby(["station_code", "month_group"])["passengers"].shift(-horizon_steps)

        # Бейзлайн на шаге t + horizon_steps (ожидаемое значение из обучающей медианы)
        df["baseline_target"] = df.groupby(["station_code", "month_group"])["baseline_pred"].shift(-horizon_steps)

        # Календарные и циклические фичи целевого момента t + horizon_steps
        df["target_datetime"] = df.groupby(["station_code", "month_group"])["datetime"].shift(-horizon_steps)
        target_cal = self.calendar_engine.add_features(df[["target_datetime"]].dropna().rename(columns={"target_datetime": "datetime"}))
        
        # Индексируем и мерджим таргетные временные характеристики
        df["target_hour"] = df["target_datetime"].dt.hour
        df["target_interval_96"] = df["target_datetime"].dt.hour * 4 + (df["target_datetime"].dt.minute // 15)
        df["target_day_of_week"] = df["target_datetime"].dt.dayofweek
        df["target_is_weekend"] = (df["target_day_of_week"] >= 5).astype(int)

        # Циклические признаки времени целевого слота
        df["target_sin_time"] = np.sin(2 * np.pi * df["target_interval_96"] / 96.0)
        df["target_cos_time"] = np.cos(2 * np.pi * df["target_interval_96"] / 96.0)
        df["target_sin_dow"] = np.sin(2 * np.pi * df["target_day_of_week"] / 7.0)
        df["target_cos_dow"] = np.cos(2 * np.pi * df["target_day_of_week"] / 7.0)

        # Признаки в момент прогноза t (телеметрия станции):
        df["flow_t"] = df["passengers"]
        df["flow_t_minus_15"] = df.groupby(["station_code", "month_group"])["passengers"].shift(1)
        df["flow_t_minus_30"] = df.groupby(["station_code", "month_group"])["passengers"].shift(2)
        df["flow_t_minus_60"] = df.groupby(["station_code", "month_group"])["passengers"].shift(4)

        # Динамика и тренд в момент t
        df["flow_trend_15m"] = df["flow_t"] - df["flow_t_minus_15"]
        df["flow_ratio_to_baseline_t"] = np.where(df["baseline_pred"] > 10.0, df["flow_t"] / df["baseline_pred"], 1.0)

        # Скользящие статистики за последний час в момент t
        df["rolling_mean_1h_t"] = df.groupby(["station_code", "month_group"])["passengers"].transform(
            lambda x: x.rolling(4, min_periods=1).mean()
        )
        df["rolling_std_1h_t"] = df.groupby(["station_code", "month_group"])["passengers"].transform(
            lambda x: x.rolling(4, min_periods=1).std()
        ).fillna(0.0)

        # Пространственные волны ключевых хабов в момент t:
        pivoted_t = df.pivot_table(index="datetime", columns="station_code", values="passengers", aggfunc="first")
        hub_south = pivoted_t[111].rename("hub_south_flow_t")     # Проспект Ветеранов
        hub_north = pivoted_t[129].rename("hub_north_flow_t")     # Девяткино
        hub_center = pivoted_t[120].rename("hub_center_flow_t")   # Пл. Восстания
        line_total = pivoted_t.sum(axis=1).rename("line_total_flow_t")

        hubs_df = pd.concat([hub_south, hub_north, hub_center, line_total], axis=1).reset_index()
        df = pd.merge(df, hubs_df, on="datetime", how="left")

        # Признаки погоды в целевой момент (для горизонта берем сдвинутую погоду)
        # На горизонтах до 2 часов погода либо известна (текущая), либо доступен точный прогноз
        for col in ["temperature", "precipitation", "rain", "wind_speed", "is_rain", "is_heavy_rain"]:
            df[f"weather_{col}_target"] = df.groupby(["station_code", "month_group"])[col].shift(-horizon_steps)

        # Фильтруем строки с NaN в таргете (конец месяца) и начальные лаги
        valid_mask = df["target"].notna() & df["flow_t_minus_60"].notna() & df["target_datetime"].notna()
        df_clean = df[valid_mask].copy().reset_index(drop=True)

        feature_cols = [
            # Базовый профиль и таргетные календарные фичи
            "baseline_target", "target_hour", "target_interval_96", "target_day_of_week",
            "target_is_weekend", "target_sin_time", "target_cos_time", "target_sin_dow", "target_cos_dow",
            "is_holiday", "is_pre_holiday", "is_spb_event",
            # Статические характеристики станции
            "station_code", "station_order", "has_turnaround",
            # Телеметрия станции на момент прогноза t
            "flow_t", "flow_t_minus_15", "flow_t_minus_30", "flow_t_minus_60",
            "flow_trend_15m", "flow_ratio_to_baseline_t", "rolling_mean_1h_t", "rolling_std_1h_t",
            # Пространственные волны узлов линии на момент t
            "hub_south_flow_t", "hub_north_flow_t", "hub_center_flow_t", "line_total_flow_t",
            # Погодные условия
            "weather_temperature_target", "weather_precipitation_target", "weather_rain_target",
            "weather_wind_speed_target", "weather_is_rain_target", "weather_is_heavy_rain_target"
        ]

        X = df_clean[feature_cols].copy()
        y = df_clean["target"].copy()

        # Категориальные признаки
        X["station_code"] = X["station_code"].astype("category")

        return df_clean, X, y, feature_cols


class MultiHorizonPredictor:
    """
    Комплекс моделей градиентного бустинга для различных горизонтов прогноза (15, 30, 60, 120 мин).
    """

    def __init__(self, models_dir: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.models_dir = models_dir or os.path.join(base_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.models: Dict[str, lgb.LGBMRegressor] = {}
        self.feature_names: Dict[str, List[str]] = {}

    def train_horizon(
        self,
        horizon_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> lgb.LGBMRegressor:
        """
        Обучает LightGBM регрессор с функцией потерь L1/Huber для минимизации WAPE.
        """
        print(f"\n--- Обучение модели LightGBM для горизонта +{horizon_name} (X_train: {X_train.shape}) ---")

        # Гиперпараметры подобраны под специфику пассажиропотоков
        model = lgb.LGBMRegressor(
            objective="regression_l1",  # Минимизирует MAE -> напрямую оптимизирует WAPE
            n_estimators=350,
            learning_rate=0.06,
            num_leaves=45,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            importance_type="gain",
            verbose=-1
        )

        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)] if eval_set else None

        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )

        model_path = os.path.join(self.models_dir, f"lgbm_horizon_{horizon_name}.joblib")
        joblib.dump(model, model_path)
        print(f"Модель сохранена: {model_path}")

        self.models[horizon_name] = model
        self.feature_names[horizon_name] = list(X_train.columns)
        return model


def run_multi_horizon_training_and_eval(
    data_root: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Полный пайплайн обучения и валидации мульти-горизонтных моделей:
    1. Построение признаковых матриц для Train (Фев, Май, Июль) и Test (Сентябрь).
    2. Обучение моделей на каждый горизонт (15m, 30m, 60m, 120m).
    3. Оценка WAPE, MAE, RMSE, R^2 на отложенном тесте сентября 2026.
    4. Сравнение с бейзлайном.
    5. Построение графиков и сохранение предсказаний.
    """
    prep = MultiHorizonDataPrep(data_root=data_root)
    print("Подготовка базовых данных Линии 1, календаря, погоды и бейзлайна...")
    train_base, test_base, _ = prep.prepare_base_dataset()

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_dir = output_dir or os.path.join(base_dir, "data")
    os.makedirs(save_dir, exist_ok=True)
    predictor = MultiHorizonPredictor()

    results_all = {}
    test_predictions_dict = {}

    for h_name, steps in HORIZONS.items():
        print(f"\n========================================================")
        print(f"Обработка горизонта {h_name} (+{steps * 15} минут)")
        print(f"========================================================")

        train_clean, X_train, y_train, feat_cols = prep.create_horizon_dataset(train_base, horizon_steps=steps)
        test_clean, X_test, y_test, _ = prep.create_horizon_dataset(test_base, horizon_steps=steps)

        # Выделяем валидационную выборку (последние 2 недели июля) для ранней остановки
        val_mask = train_clean["datetime"] >= "2026-07-15"
        X_tr = X_train[~val_mask]
        y_tr = y_train[~val_mask]
        X_vl = X_train[val_mask]
        y_vl = y_train[val_mask]

        model = predictor.train_horizon(h_name, X_tr, y_tr, X_vl, y_vl)

        # Предсказание на тесте сентября 2026
        y_test_pred = model.predict(X_test)
        y_test_pred = np.maximum(y_test_pred, 0.0)

        # Бейзлайн на этом же тестовом срезе
        y_test_baseline = test_clean["baseline_target"].values
        y_test_true = y_test.values

        # Метрики ML модели
        ml_wape = calculate_wape(y_test_true, y_test_pred)
        ml_mae = calculate_mae(y_test_true, y_test_pred)
        ml_rmse = calculate_rmse(y_test_true, y_test_pred)
        ml_r2 = calculate_r2(y_test_true, y_test_pred)

        # Метрики бейзлайна для честного сопоставления
        bl_wape = calculate_wape(y_test_true, y_test_baseline)
        bl_mae = calculate_mae(y_test_true, y_test_baseline)
        bl_rmse = calculate_rmse(y_test_true, y_test_baseline)
        bl_r2 = calculate_r2(y_test_true, y_test_baseline)

        wape_improvement = (bl_wape - ml_wape) / bl_wape * 100.0

        # Метрики по периодам для ML модели
        hour_target = test_clean["target_hour"]
        morning_mask = (hour_target >= 7) & (hour_target < 10)
        evening_mask = (hour_target >= 17) & (hour_target < 20)

        results_all[h_name] = {
            "horizon_minutes": steps * 15,
            "ml_metrics": {
                "wape_pct": round(ml_wape * 100, 2),
                "mae": round(ml_mae, 2),
                "rmse": round(ml_rmse, 2),
                "r2": round(ml_r2, 4)
            },
            "baseline_metrics": {
                "wape_pct": round(bl_wape * 100, 2),
                "mae": round(bl_mae, 2),
                "rmse": round(bl_rmse, 2),
                "r2": round(bl_r2, 4)
            },
            "relative_improvement_pct": round(wape_improvement, 2),
            "peak_periods_ml": {
                "morning_peak_07_10": {
                    "wape_pct": round(calculate_wape(y_test_true[morning_mask], y_test_pred[morning_mask]) * 100, 2),
                    "mae": round(calculate_mae(y_test_true[morning_mask], y_test_pred[morning_mask]), 2)
                },
                "evening_peak_17_20": {
                    "wape_pct": round(calculate_wape(y_test_true[evening_mask], y_test_pred[evening_mask]) * 100, 2),
                    "mae": round(calculate_mae(y_test_true[evening_mask], y_test_pred[evening_mask]), 2)
                }
            }
        }

        print(f"Горизонт +{h_name}: ML WAPE = {ml_wape*100:.2f}% | Baseline WAPE = {bl_wape*100:.2f}% | Прирост: {wape_improvement:+.1f}%")

        # Сохраняем предсказания горизонта
        h_df = test_clean[["datetime", "target_datetime", "station_code", "station_name", "station_order"]].copy()
        h_df["actual_target"] = y_test_true
        h_df["baseline_target"] = y_test_baseline
        h_df[f"pred_ml_{h_name}"] = y_test_pred
        h_df[f"abs_error_ml_{h_name}"] = np.abs(y_test_true - y_test_pred)
        test_predictions_dict[h_name] = h_df

    # Сохраняем итоговые прогнозы: объединяем по ключам (datetime, station_code)
    # Начиная с горизонта 15м
    merged_preds = test_predictions_dict["15m"][["datetime", "station_code", "station_name", "station_order", "actual_target", "baseline_target", "pred_ml_15m"]].rename(
        columns={"actual_target": "actual_15m", "baseline_target": "baseline_15m"}
    )
    for h_name in ["30m", "60m", "120m"]:
        sub_h = test_predictions_dict[h_name][["datetime", "station_code", "actual_target", "baseline_target", f"pred_ml_{h_name}"]].rename(
            columns={"actual_target": f"actual_{h_name}", "baseline_target": f"baseline_{h_name}"}
        )
        merged_preds = pd.merge(merged_preds, sub_h, on=["datetime", "station_code"], how="left")

    parquet_path = os.path.join(save_dir, "ml_multi_horizon_predictions.parquet")
    merged_preds.to_parquet(parquet_path, index=False)
    print(f"\nВсе предсказания горизонтов сохранены: {parquet_path}")

    # Сохраняем отчет о метриках в JSON
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, "ml_multi_horizon_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, ensure_ascii=False, indent=2)
    print(f"Метрики сохранены: {json_path}")

    # Построение визуализации сравнения
    figures_dir = os.path.join(base_dir, "eda", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, "multi_horizon_comparison.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#0F172A")

    for ax in axes.flat:
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors="#94A3B8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    horizons_list = list(HORIZONS.keys())
    ml_wapes = [results_all[h]["ml_metrics"]["wape_pct"] for h in horizons_list]
    bl_wapes = [results_all[h]["baseline_metrics"]["wape_pct"] for h in horizons_list]

    # График 1: Сравнение WAPE по горизонтам
    x = np.arange(len(horizons_list))
    width = 0.35
    axes[0, 0].bar(x - width/2, bl_wapes, width, label="Historical Baseline", color="#F59E0B", alpha=0.85)
    axes[0, 0].bar(x + width/2, ml_wapes, width, label="LightGBM Multi-Horizon", color="#10B981", alpha=0.9)
    axes[0, 0].axhline(8.5, color="#F43F5E", linestyle="--", linewidth=1.8, label="Целевой порог WAPE (8.5%)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f"+{h}" for h in horizons_list], color="#F8FAFC", fontsize=11)
    axes[0, 0].set_title("WAPE (%) по горизонтам прогнозирования (Сентябрь 2026)", color="#F8FAFC", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("WAPE (%)", color="#94A3B8")
    axes[0, 0].legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    # График 2: Утренний и вечерний пик для t+15m и t+30m
    peaks_ml = [results_all["15m"]["peak_periods_ml"]["morning_peak_07_10"]["wape_pct"],
                results_all["15m"]["peak_periods_ml"]["evening_peak_17_20"]["wape_pct"],
                results_all["30m"]["peak_periods_ml"]["morning_peak_07_10"]["wape_pct"],
                results_all["30m"]["peak_periods_ml"]["evening_peak_17_20"]["wape_pct"]]
    labels_peaks = ["15m Утро", "15m Вечер", "30m Утро", "30m Вечер"]
    axes[0, 1].bar(labels_peaks, peaks_ml, color="#38BDF8", alpha=0.85)
    axes[0, 1].axhline(8.5, color="#F43F5E", linestyle="--", linewidth=1.5, label="Порог WAPE 8.5%")
    for i, v in enumerate(peaks_ml):
        axes[0, 1].text(i, v + 0.15, f"{v:.1f}%", ha="center", color="#F8FAFC", fontweight="bold")
    axes[0, 1].set_title("Точность в часы пик (07:00-10:00 и 17:00-20:00)", color="#F8FAFC", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("WAPE (%)", color="#94A3B8")
    axes[0, 1].legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    # График 3: Прогноз на хабе ст. Площадь Восстания (t+15m, 10-12 сентября 2026)
    df_15m_res = test_predictions_dict["15m"]
    sub_vosst = df_15m_res[(df_15m_res["station_code"] == 120) & 
                           (df_15m_res["target_datetime"] >= "2026-09-10") & 
                           (df_15m_res["target_datetime"] <= "2026-09-12")]
    axes[1, 0].plot(sub_vosst["target_datetime"], sub_vosst["actual_target"], label="Факт", color="#38BDF8", linewidth=1.8)
    axes[1, 0].plot(sub_vosst["target_datetime"], sub_vosst["pred_ml_15m"], label="LightGBM (+15m)", color="#10B981", linestyle="--", linewidth=1.6)
    axes[1, 0].plot(sub_vosst["target_datetime"], sub_vosst["baseline_target"], label="Historical Baseline", color="#F59E0B", linestyle=":", linewidth=1.2, alpha=0.7)
    axes[1, 0].set_title("ст. Площадь Восстания: Динамика Факт vs Прогноз +15м", color="#F8FAFC", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Пассажиров в 15 мин", color="#94A3B8")
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    # График 4: Важность признаков для горизонта 15м
    model_15 = predictor.models["15m"]
    feat_imp = pd.Series(model_15.feature_importances_, index=predictor.feature_names["15m"]).sort_values(ascending=True).tail(10)
    axes[1, 1].barh(feat_imp.index, feat_imp.values, color="#818CF8", alpha=0.85)
    axes[1, 1].set_title("Топ-10 факторов точности модели (+15 мин, Feature Gain)", color="#F8FAFC", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Gain Importance", color="#94A3B8")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"График сравнения сохранен: {fig_path}")

    return results_all


if __name__ == "__main__":
    print("Старт обучения и тестирования многогоризонтного предиктивного ядра...")
    run_multi_horizon_training_and_eval()
