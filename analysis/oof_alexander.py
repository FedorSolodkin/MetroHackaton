"""Прогнозы модели Александра (остаточный LightGBM, P50, горизонт 30 мин, его признаки и параметры) «месяц в проверку»,
построчно: (время цели, станция) -> прогноз. Сохраняет ../oof_alexander.pkl. Запуск из корня репозитория."""
import os, sys, time, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, lightgbm as lgb
from src.data.data_loader import MetroDataLoader
from src.models.baseline import MetroBaselineModel
from src.models.feature_builder import MetroFeatureBuilder, FEATURE_COLUMNS, CATEGORICAL_FEATURES, STATION_METADATA
from src.models.train_multi_horizon import MultiHorizonFlowModel

t0 = time.time(); L = MetroDataLoader(); dfs = L.get_station_aggregated_flow(L.load_15min_dataset()); dfs["month"] = pd.to_datetime(dfs["datetime"]).dt.month
params = MultiHorizonFlowModel()._get_lgb_params(0.5, 30); out = []
for m in (2, 5, 7, 9):
    tr_raw, val_raw = dfs[dfs.month != m].copy(), dfs[dfs.month == m].copy()
    fb = MetroBaselineModel(); fb.fit(tr_raw); bld = MetroFeatureBuilder(baseline_model=fb)
    tr_df, fc = bld.build_training_dataset_for_horizon(tr_raw, horizon_min=30); val_df, _ = bld.build_training_dataset_for_horizon(val_raw, horizon_min=30)
    bst = lgb.train(params, lgb.Dataset(tr_df[fc], label=tr_df["target_res"], categorical_feature=CATEGORICAL_FEATURES), num_boost_round=120)
    p = np.maximum(0.0, val_df["base_pax_target"].values + bst.predict(val_df[fc]))
    out.append(pd.DataFrame({"target_datetime": val_df["target_datetime"].values, "station": val_df["station_code"].map(lambda c: STATION_METADATA[c]["station_name"]).values,
                             "y": val_df["target_pax"].values, "pred_alex": p, "base_alex": val_df["base_pax_target"].values}))
    print(f"месяц {m}: {len(val_df):,} строк, WAPE {np.abs(val_df.target_pax.values-p).sum()/val_df.target_pax.values.sum()*100:.2f}%, {time.time()-t0:.0f} с", flush=True)
res = pd.concat(out); pickle.dump(res, open(os.path.join(os.path.dirname(ROOT), "oof_alexander.pkl"), "wb")); print("сохранено", len(res))
