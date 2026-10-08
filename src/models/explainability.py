"""
Модуль интерпретируемости предиктивных моделей (XAI / SHAP) для ЦУП Петербургского метрополитена.
Объясняет диспетчеру причины предиктивных всплесков пассажиропотока:
- Влияние погодных аномалий (ливни, похолодание)
- Пространственные волны от узловых хабов (Девяткино, Проспект Ветеранов, Пл. Восстания)
- Календарные сдвиги (сокращенные предпраздничные дни)
"""

from typing import Dict, Any, List, Optional
import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.train_multi_horizon import MultiHorizonDataPrep, HORIZONS
from configs.stations import get_station_info


# Человекочитаемые названия фичей для диспетчеров метрополитена
FEATURE_TRANSLATIONS = {
    "baseline_target": "Исторический норматив (типовой профиль)",
    "line_total_flow_t": "Общий приток по всей Линии 1 сейчас",
    "flow_t": "Текущий приток на станции (t)",
    "hub_south_flow_t": "Волна с южного хаба (Пр. Ветеранов)",
    "hub_north_flow_t": "Волна с северного хаба (Девяткино)",
    "hub_center_flow_t": "Нагрузка центрального узла (Пл. Восстания)",
    "flow_trend_15m": "Динамика притока за 15 мин (тренд)",
    "flow_ratio_to_baseline_t": "Коэф. отклонения от нормы сейчас",
    "target_interval_96": "Время суток (15-мин слот)",
    "target_hour": "Час суток",
    "target_day_of_week": "День недели",
    "target_is_weekend": "Выходной день",
    "is_pre_holiday": "Предпраздничный день (ранний пик)",
    "is_holiday": "Праздничный день",
    "is_spb_event": "Массовое городское событие в СПб",
    "weather_temperature_target": "Температура воздуха (°C)",
    "weather_rain_target": "Интенсивность дождя (мм/ч)",
    "weather_precipitation_target": "Осадки (мм)",
    "weather_wind_speed_target": "Скорость ветра (м/с)",
    "weather_is_heavy_rain_target": "Штормовой ливень",
    "weather_is_rain_target": "Дождливая погода",
    "rolling_mean_1h_t": "Средний приток за последний час",
    "rolling_std_1h_t": "Нестабильность притока (дисперсия)",
    "station_code": "Станция",
    "station_order": "Порядковый номер станции",
    "has_turnaround": "Наличие путевого развития (тупик)",
}


class DispatchExplainer:
    """
    Класс генерации SHAP-объяснений в реальном времени для диспетчерского пульта ЦУП.
    """

    def __init__(self, horizon: str = "15m", models_dir: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.models_dir = models_dir or os.path.join(base_dir, "models")
        model_path = os.path.join(self.models_dir, f"lgbm_horizon_{horizon}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель {model_path} не найдена. Сначала выполните обучение train_multi_horizon.py")

        self.model = joblib.load(model_path)
        self.horizon = horizon
        self.feature_names = self.model.feature_name_
        self.explainer = shap.TreeExplainer(self.model)

    def explain_observation(
        self,
        row_df: pd.DataFrame,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Формирует лаконичный структурированный отчет для диспетчера:
        почему прогноз отличается от базового норматива и какие внешние факторы стали драйверами.
        """
        X = row_df[self.feature_names].copy()
        pred_val = float(self.model.predict(X)[0])
        shap_values = self.explainer.shap_values(X)[0]
        base_val = float(self.explainer.expected_value)

        # Сортируем по абсолютному вкладу
        impacts = []
        for feat, val, sh in zip(self.feature_names, X.iloc[0].values, shap_values):
            human_name = FEATURE_TRANSLATIONS.get(feat, feat)
            impacts.append({
                "feature": feat,
                "human_name": human_name,
                "feature_value": float(val) if isinstance(val, (int, float, np.number)) else str(val),
                "shap_impact": float(sh),
                "abs_impact": abs(float(sh))
            })

        impacts.sort(key=lambda x: x["abs_impact"], reverse=True)
        top_impacts = impacts[:top_k]

        # Генерируем естественные текстовые формулировки
        text_reasons = []
        for imp in top_impacts:
            sign = "+" if imp["shap_impact"] > 0 else "-"
            sh_round = abs(round(imp["shap_impact"]))
            if "rain" in imp["feature"] or "precipitation" in imp["feature"]:
                text_reasons.append(f"Погодный фактор: осадки ({sign}{sh_round} пасс/15мин)")
            elif "hub" in imp["feature"]:
                text_reasons.append(f"Волна с пересадочного узла: {imp['human_name']} ({sign}{sh_round} пасс/15мин)")
            elif "line_total" in imp["feature"]:
                text_reasons.append(f"Общесетевая динамика Линии 1 ({sign}{sh_round} пасс/15мин)")
            elif "trend" in imp["feature"]:
                text_reasons.append(f"Локальный разгон пассажиропотока ({sign}{sh_round} пасс/15мин)")
            else:
                text_reasons.append(f"{imp['human_name']} ({sign}{sh_round} пасс/15мин)")

        return {
            "predicted_flow": round(pred_val, 1),
            "base_expected_value": round(base_val, 1),
            "top_drivers": top_impacts,
            "dispatcher_reasons": text_reasons
        }


def generate_shap_report(
    data_root: Optional[str] = None,
    output_dir: Optional[str] = None
) -> None:
    """
    Генерирует глобальные графики SHAP (beeswarm, waterfall, bar)
    для презентации и аналитического отчета.
    """
    print("Инициализация SHAP Explainability для горизонта +15m...")
    prep = MultiHorizonDataPrep(data_root=data_root)
    _, test_base, _ = prep.prepare_base_dataset()
    _, X_test, y_test, feat_cols = prep.create_horizon_dataset(test_base, horizon_steps=1)

    explainer_tool = DispatchExplainer(horizon="15m")
    
    # Берем репрезентативную подвыборку тестов для SHAP (1200 точек для высокой скорости и точности)
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), size=min(1200, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_indices].copy()

    print("Расчет значений SHAP TreeExplainer...")
    shap_values = explainer_tool.explainer.shap_values(X_sample)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    figures_dir = output_dir or os.path.join(base_dir, "eda", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Построение Global Summary Plot (Beeswarm)
    fig_summary_path = os.path.join(figures_dir, "shap_summary_h15.png")
    plt.figure(figsize=(12, 8))
    plt.gcf().patch.set_facecolor("#0F172A")
    ax = plt.gca()
    ax.set_facecolor("#1E293B")
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # Создаем DataFrame с переведенными именами колонок для понятного графика
    X_sample_ru = X_sample.copy()
    X_sample_ru.columns = [FEATURE_TRANSLATIONS.get(c, c) for c in X_sample_ru.columns]

    shap.summary_plot(
        shap_values,
        X_sample_ru,
        show=False,
        max_display=12,
        plot_size=(12, 8),
        color_bar_label="Значение признака (Низкое -> Высокое)"
    )
    plt.title("SHAP Feature Importance (+15 мин, Линия 1 СПб Метро)", color="#F8FAFC", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(fig_summary_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"SHAP Summary Plot сохранен: {fig_summary_path}")

    # 2. Построение Waterfall Plot для реального инцидента (вечерний пик на ст. Площадь Восстания)
    fig_waterfall_path = os.path.join(figures_dir, "shap_case_study.png")
    # Находим строку с максимальным притоком на ст. 120 (Пл. Восстания) в сентябре
    vosst_mask = (X_test["station_code"] == 120) & (X_test["target_hour"] == 18)
    if vosst_mask.sum() > 0:
        idx_case = X_test[vosst_mask]["flow_t"].idxmax()
        row_case = X_test.loc[[idx_case]]
        
        explanation = shap.Explanation(
            values=explainer_tool.explainer.shap_values(row_case)[0],
            base_values=explainer_tool.explainer.expected_value,
            data=row_case.iloc[0].values,
            feature_names=[FEATURE_TRANSLATIONS.get(c, c) for c in row_case.columns]
        )

        plt.figure(figsize=(10, 6))
        plt.gcf().patch.set_facecolor("#0F172A")
        shap.plots.waterfall(explanation, max_display=8, show=False)
        plt.title("Разбор решения СППР: Пиковый всплеск на ст. Площадь Восстания", color="#F8FAFC", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(fig_waterfall_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"SHAP Waterfall Plot сохранен: {fig_waterfall_path}")

    # 3. Демонстрация работы API интерпретатора для диспетчера
    test_sample = X_test.iloc[[0]]
    demo_explanation = explainer_tool.explain_observation(test_sample)
    print("\n--- ДЕМОНСТРАЦИЯ КАРТОЧКИ ИНТЕРПРЕТАЦИИ ДЛЯ ДИСПЕТЧЕРА ---")
    print(f"Прогноз притока: {demo_explanation['predicted_flow']} пасс/15мин")
    print("Ключевые драйверы прогноза:")
    for reason in demo_explanation["dispatcher_reasons"]:
        print(f"  • {reason}")


if __name__ == "__main__":
    generate_shap_report()
