"""
Модуль обучения мультигоризонтных моделей LightGBM для Линии 1 («МетроПульс-1»).
Задачи F2, F3, F4:
- Горизонты прогнозирования: 15, 30, 60, 120 минут (F2)
- Моделирование аномального остатка от базового профиля (Target: Delta = pax - base) (F3)
- Квантильная регрессия риска P10 / P50 / P90 (F4)
- 4-фолдовая Leave-One-Month-Out кросс-валидация и калибровка квантилей
- Сохранение обученных моделей в models/multi_horizon_models.joblib
"""

from __future__ import annotations

import os
import json
import time
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.linear_model import Ridge

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.baseline import MetroBaselineModel, compute_wape, compute_mae
from src.models.feature_builder import MetroFeatureBuilder, FEATURE_COLUMNS, CATEGORICAL_FEATURES


SUPPORTED_HORIZONS = [15, 30, 60, 120]
QUANTILES = [0.10, 0.50, 0.90]


class MultiHorizonFlowModel:
    """
    Мультигоризонтный ансамбль квантильных моделей LightGBM + Ridge на остатках пассажиропотока.
    Для каждого горизонта H in [15, 30, 60, 120] хранит квантильные бустеры (P10, P50, P90)
    и линейную модель Ridge для устойчивого блендинга (70% LightGBM + 30% Ridge).
    """

    def __init__(
        self,
        baseline_model: Optional[MetroBaselineModel] = None,
        feature_builder: Optional[MetroFeatureBuilder] = None,
        horizons: Optional[List[int]] = None
    ):
        self.horizons = horizons or SUPPORTED_HORIZONS
        self.baseline_model = baseline_model or MetroBaselineModel()
        self.feature_builder = feature_builder or MetroFeatureBuilder(self.baseline_model)
        
        # Словарь моделей: models[H][alpha] -> lgb.Booster
        self.models: Dict[int, Dict[float, lgb.Booster]] = {}
        # Линейные модели Ridge на остатках: ridge_models[H] -> Ridge
        self.ridge_models: Dict[int, Ridge] = {}
        # Прямые модели для сравнения (Direct): direct_models[H] -> lgb.Booster
        self.direct_models: Dict[int, lgb.Booster] = {}
        self.feature_cols = FEATURE_COLUMNS
        self.cat_cols = CATEGORICAL_FEATURES
        self.metadata: Dict[str, Any] = {}

    def _get_lgb_params(self, alpha: float, horizon_min: int) -> Dict[str, Any]:
        """Возвращает адаптированные гиперпараметры LightGBM для квантилей и горизонтов."""
        # Базовые параметры
        params = {
            "objective": "quantile",
            "alpha": alpha,
            "boosting_type": "gbdt",
            "learning_rate": 0.06 if horizon_min <= 30 else 0.04,
            "num_leaves": 31 if horizon_min <= 30 else 24,
            "min_data_in_leaf": 50 if horizon_min <= 30 else 80,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "verbosity": -1,
            "num_threads": 4,
            "seed": 42
        }
        
        # На дальних горизонтах (60-120 мин) усиливаем регуляризацию к базовому медианному якорю
        if horizon_min >= 60:
            params["reg_alpha"] = 0.5
            params["reg_lambda"] = 1.0
        if horizon_min >= 120:
            params["reg_alpha"] = 1.0
            params["reg_lambda"] = 2.0
            
        return params

    def fit(self, df_station: pd.DataFrame, num_boost_round: int = 150) -> MultiHorizonFlowModel:
        """
        Обучает квантильные модели (P10, P50, P90) для всех горизонтов на всех доступных данных.
        """
        if self.baseline_model.profile_df is None:
            self.baseline_model.fit(df_station)

        for H in self.horizons:
            dataset_h, feat_cols = self.feature_builder.build_training_dataset_for_horizon(df_station, horizon_min=H)
            self.models[H] = {}
            
            dtrain_res = lgb.Dataset(
                dataset_h[feat_cols],
                label=dataset_h["target_res"],
                categorical_feature=self.cat_cols,
                free_raw_data=False
            )
            
            for q in QUANTILES:
                params = self._get_lgb_params(alpha=q, horizon_min=H)
                booster = lgb.train(params, dtrain_res, num_boost_round=num_boost_round)
                self.models[H][q] = booster

            # 2. Линейная модель Ridge для устойчивого блендинга
            ridge = Ridge(alpha=100.0, random_state=42)
            X_tr = dataset_h[feat_cols].fillna(0.0)
            ridge.fit(X_tr, dataset_h["target_res"])
            self.ridge_models[H] = ridge

        self.metadata = {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "horizons": self.horizons,
            "quantiles": QUANTILES,
            "feature_columns": self.feature_cols,
            "training_samples": len(df_station),
            "architecture": "Blend (70% LightGBM + 30% Ridge)"
        }
        return self

    def predict_horizon(
        self,
        inference_features_df: pd.DataFrame,
        horizon_min: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычисляет прогноз (pred_p10, pred_p50, pred_p90) для станций на заданный горизонт.
        Использует робастный бленд 70% LightGBM + 30% Ridge.
        Гарантирует монотонность квантилей: pred_p10 <= pred_p50 <= pred_p90 и неотрицательность потока.
        """
        if horizon_min not in self.models:
            raise ValueError(f"Горизонт {horizon_min} мин не обучен. Доступные: {list(self.models.keys())}")

        X = inference_features_df[self.feature_cols]
        base_target = inference_features_df["base_pax_target"].values

        pred_res_p10 = self.models[horizon_min][0.10].predict(X)
        pred_res_p50_lgb = self.models[horizon_min][0.50].predict(X)
        pred_res_p90 = self.models[horizon_min][0.90].predict(X)

        # Применяем бленд с Ridge моделью
        if horizon_min in self.ridge_models:
            pred_res_ridge = self.ridge_models[horizon_min].predict(X.fillna(0.0))
            pred_res_p50 = 0.70 * pred_res_p50_lgb + 0.30 * pred_res_ridge
            delta_shift = pred_res_p50 - pred_res_p50_lgb
            pred_res_p10 = pred_res_p10 + 0.30 * delta_shift
            pred_res_p90 = pred_res_p90 + 0.30 * delta_shift
        else:
            pred_res_p50 = pred_res_p50_lgb

        # Переход от остатков к абсолютным пассажирам
        p10 = np.maximum(0.0, base_target + pred_res_p10)
        p50 = np.maximum(0.0, base_target + pred_res_p50)
        p90 = np.maximum(0.0, base_target + pred_res_p90)

        # Гарантия монотонности: p10 <= p50 <= p90
        p50 = np.maximum(p10, p50)
        p90 = np.maximum(p50, p90)

        return p10, p50, p90

    def evaluate_lomo(
        self,
        df_station: pd.DataFrame,
        months: Tuple[int, ...] = (2, 5, 7, 9),
        num_boost_round: int = 120
    ) -> Dict[str, Any]:
        """
        Честная 4-фолдовая Leave-One-Month-Out кросс-валидация:
        - Обучение моделей на 3 месяцах, валидация на 4-м
        - Сравнение: Baseline vs Direct LightGBM vs Residual LightGBM
        - Калибровка квантилей: факт > P90 (ожидание ~10%), факт < P10 (ожидание ~10%)
        - Ошибка на пиках (07:30-10:00, 17:00-19:30) и на аномальных слотах (|res| > 120)
        """
        results_by_horizon = {}
        df_clean = df_station.copy()
        df_clean["month"] = pd.to_datetime(df_clean["datetime"]).dt.month

        for H in self.horizons:
            fold_metrics = []
            
            for val_m in months:
                tr_raw = df_clean[df_clean["month"] != val_m].copy()
                val_raw = df_clean[df_clean["month"] == val_m].copy()
                
                # Строго Zero Leakage: базовый профиль обучается только на 3 тренировочных месяцах фолда!
                fold_base = MetroBaselineModel(calendar_path=self.baseline_model.calendar_path)
                fold_base.fit(tr_raw)
                fold_builder = MetroFeatureBuilder(baseline_model=fold_base)
                
                tr_df, feat_cols = fold_builder.build_training_dataset_for_horizon(tr_raw, horizon_min=H)
                val_df, _ = fold_builder.build_training_dataset_for_horizon(val_raw, horizon_min=H)
                
                # 1. Baseline прогноз на валидационном месяце
                y_true = val_df["target_pax"].values
                y_base = val_df["base_pax_target"].values
                wape_base = compute_wape(y_true, y_base)
                mae_base = compute_mae(y_true, y_base)
                
                # 2. Прямая модель (Direct LightGBM)
                dtrain_direct = lgb.Dataset(
                    tr_df[feat_cols],
                    label=tr_df["target_pax"],
                    categorical_feature=self.cat_cols,
                    free_raw_data=False
                )
                params_direct = self._get_lgb_params(alpha=0.50, horizon_min=H)
                booster_direct = lgb.train(params_direct, dtrain_direct, num_boost_round=num_boost_round)
                pred_direct = np.maximum(0.0, booster_direct.predict(val_df[feat_cols]))
                wape_direct = compute_wape(y_true, pred_direct)
                mae_direct = compute_mae(y_true, pred_direct)
                
                # 3. Остаточная квантильная модель (Residual LightGBM)
                dtrain_res = lgb.Dataset(
                    tr_df[feat_cols],
                    label=tr_df["target_res"],
                    categorical_feature=self.cat_cols,
                    free_raw_data=False
                )
                
                booster_p10 = lgb.train(self._get_lgb_params(0.10, H), dtrain_res, num_boost_round=num_boost_round)
                booster_p50 = lgb.train(self._get_lgb_params(0.50, H), dtrain_res, num_boost_round=num_boost_round)
                booster_p90 = lgb.train(self._get_lgb_params(0.90, H), dtrain_res, num_boost_round=num_boost_round)
                
                pred_res_p10 = np.maximum(0.0, y_base + booster_p10.predict(val_df[feat_cols]))
                pred_res_p50 = np.maximum(0.0, y_base + booster_p50.predict(val_df[feat_cols]))
                pred_res_p90 = np.maximum(0.0, y_base + booster_p90.predict(val_df[feat_cols]))
                
                # Монотонность
                pred_res_p50 = np.maximum(pred_res_p10, pred_res_p50)
                pred_res_p90 = np.maximum(pred_res_p50, pred_res_p90)
                
                wape_residual = compute_wape(y_true, pred_res_p50)
                mae_residual = compute_mae(y_true, pred_res_p50)
                
                # Калибровка
                cov_p90 = float(np.mean(y_true > pred_res_p90))
                cov_p10 = float(np.mean(y_true < pred_res_p10))
                
                # Ошибка на пиках (слоты 30..40 = 07:30-10:00 и слоты 68..78 = 17:00-19:30)
                slots = val_df["target_slot_index"].values
                peak_mask = ((slots >= 30) & (slots <= 40)) | ((slots >= 68) & (slots <= 78))
                wape_peak_base = compute_wape(y_true[peak_mask], y_base[peak_mask]) if peak_mask.any() else 0.0
                wape_peak_res = compute_wape(y_true[peak_mask], pred_res_p50[peak_mask]) if peak_mask.any() else 0.0
                
                # Ошибка на аномальных слотах (|y - base| > 120)
                tail_mask = np.abs(y_true - y_base) > 120.0
                tail_wape_base = compute_wape(y_true[tail_mask], y_base[tail_mask]) if tail_mask.any() else 0.0
                tail_wape_res = compute_wape(y_true[tail_mask], pred_res_p50[tail_mask]) if tail_mask.any() else 0.0
                
                fold_metrics.append({
                    "val_month": val_m,
                    "wape_base": wape_base,
                    "mae_base": mae_base,
                    "wape_direct": wape_direct,
                    "mae_direct": mae_direct,
                    "wape_residual": wape_residual,
                    "mae_residual": mae_residual,
                    "cov_p90": cov_p90,
                    "cov_p10": cov_p10,
                    "wape_peak_base": wape_peak_base,
                    "wape_peak_res": wape_peak_res,
                    "tail_wape_base": tail_wape_base,
                    "tail_wape_res": tail_wape_res
                })
                
            metrics_df = pd.DataFrame(fold_metrics)
            results_by_horizon[H] = {
                "mean_wape_base": float(metrics_df["wape_base"].mean()),
                "mean_wape_direct": float(metrics_df["wape_direct"].mean()),
                "mean_wape_residual": float(metrics_df["wape_residual"].mean()),
                "mean_mae_base": float(metrics_df["mae_base"].mean()),
                "mean_mae_residual": float(metrics_df["mae_residual"].mean()),
                "mean_cov_p90": float(metrics_df["cov_p90"].mean()),
                "mean_cov_p10": float(metrics_df["cov_p10"].mean()),
                "mean_wape_peak_base": float(metrics_df["wape_peak_base"].mean()),
                "mean_wape_peak_res": float(metrics_df["wape_peak_res"].mean()),
                "mean_tail_wape_base": float(metrics_df["tail_wape_base"].mean()),
                "mean_tail_wape_res": float(metrics_df["tail_wape_res"].mean()),
                "fold_details": fold_metrics
            }

        return results_by_horizon

    def save(self, model_path: str):
        """Сохраняет модели в joblib."""
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        payload = {
            "models": self.models,
            "ridge_models": self.ridge_models,
            "horizons": self.horizons,
            "feature_cols": self.feature_cols,
            "cat_cols": self.cat_cols,
            "metadata": self.metadata
        }
        joblib.dump(payload, model_path)
        
        # Сохраняем текстовый json с описанием
        meta_json_path = os.path.splitext(model_path)[0] + "_metadata.json"
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def load(self, model_path: str) -> MultiHorizonFlowModel:
        """Загружает сохраненные модели."""
        payload = joblib.load(model_path)
        self.models = payload["models"]
        self.ridge_models = payload.get("ridge_models", {})
        self.horizons = payload.get("horizons", SUPPORTED_HORIZONS)
        self.feature_cols = payload.get("feature_cols", FEATURE_COLUMNS)
        self.cat_cols = payload.get("cat_cols", CATEGORICAL_FEATURES)
        self.metadata = payload.get("metadata", {})
        return self


if __name__ == "__main__":
    from src.data.data_loader import MetroDataLoader

    loader = MetroDataLoader()
    df_raw = loader.load_15min_dataset()
    df_station = loader.get_station_aggregated_flow(df_raw)

    print("=" * 70)
    print("МУЛЬТИГОРИЗОНТНОЕ ОБУЧЕНИЕ И LOMO ВАЛИДАЦИЯ (F2, F3, F4)")
    print("=" * 70)

    base_model = MetroBaselineModel()
    base_model.fit(df_station)
    
    mh_model = MultiHorizonFlowModel(baseline_model=base_model)
    
    print("\n[1/2] Запуск 4-фолдовой LOMO кросс-валидации по всем горизонтам...")
    eval_results = mh_model.evaluate_lomo(df_station)

    print("\n" + "=" * 70)
    print("СВОДНЫЕ РЕЗУЛЬТАТЫ LOMO КРОСС-ВАЛИДАЦИИ ПО ГОРИЗОНТАМ")
    print("=" * 70)
    print(f"{'Горизонт':<10} | {'Base WAPE':<10} | {'Direct LGB':<11} | {'Residual LGB':<13} | {'Выигрыш':<10} | {'Факт>P90':<9} | {'Факт<P10':<9}")
    print("-" * 80)
    for H in SUPPORTED_HORIZONS:
        res = eval_results[H]
        wb = res["mean_wape_base"] * 100
        wd = res["mean_wape_direct"] * 100
        wr = res["mean_wape_residual"] * 100
        gain = wb - wr
        p90 = res["mean_cov_p90"] * 100
        p10 = res["mean_cov_p10"] * 100
        print(f"{H:>4} мин    | {wb:>8.2f}% | {wd:>9.2f}% | {wr:>11.2f}% | {gain:>+8.2f} п.п. | {p90:>7.1f}% | {p10:>7.1f}%")

    print("-" * 80)
    print("\nОШИБКА НА ПИКАХ И АНОМАЛИЯХ (Tail-WAPE при |y - Base| > 120 пасс):")
    for H in SUPPORTED_HORIZONS:
        res = eval_results[H]
        tb = res["mean_tail_wape_base"] * 100
        tr = res["mean_tail_wape_res"] * 100
        gain_tail = tb - tr
        print(f"  H={H:>3} мин: Tail-WAPE База {tb:.2f}% -> Остаточный LGB {tr:.2f}% (сокращение ошибки на {gain_tail:+.2f} п.п.)")

    print("\n[2/2] Обучение финальных моделей на полном датасете и сохранение артефактов...")
    mh_model.fit(df_station)
    save_path = os.path.join(loader.data_root, "..", "models", "multi_horizon_models.joblib")
    mh_model.save(save_path)
    print(f"Модели успешно сохранены в: {save_path}")
