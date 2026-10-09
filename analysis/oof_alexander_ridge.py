"""Прогнозы Ridge Александра (Ridge(alpha=100) на остатках от его базы, его признаки), бленд 70/30 с его LightGBM, горизонт 30 мин, «месяц в проверку».
Сохраняет ../oof_alex_ridge.pkl. Запуск из корня репозитория."""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import Ridge
from src.data.data_loader import MetroDataLoader
from src.models.baseline import MetroBaselineModel
from src.models.feature_builder import MetroFeatureBuilder, CATEGORICAL_FEATURES, STATION_METADATA
from src.models.train_multi_horizon import MultiHorizonFlowModel
L = MetroDataLoader(); dfs = L.get_station_aggregated_flow(L.load_15min_dataset()); dfs["month"] = pd.to_datetime(dfs["datetime"]).dt.month
mh = MultiHorizonFlowModel(); out = []
for m in (2, 5, 7, 9):
    tr_raw, va_raw = dfs[dfs.month != m].copy(), dfs[dfs.month == m].copy(); fb = MetroBaselineModel(); fb.fit(tr_raw); bld = MetroFeatureBuilder(baseline_model=fb)
    tr, fc = bld.build_training_dataset_for_horizon(tr_raw, horizon_min=30); va, _ = bld.build_training_dataset_for_horizon(va_raw, horizon_min=30)
    bst = lgb.train(mh._get_lgb_params(0.5, 30), lgb.Dataset(tr[fc], label=tr["target_res"], categorical_feature=CATEGORICAL_FEATURES), num_boost_round=120)
    rg = Ridge(alpha=100.0, random_state=42).fit(tr[fc].fillna(0.0), tr["target_res"]); b = va["base_pax_target"].values; pl = bst.predict(va[fc]); pr = rg.predict(va[fc].fillna(0.0))
    out.append(pd.DataFrame({"target_datetime": va["target_datetime"].values, "station": va["station_code"].map(lambda c: STATION_METADATA[c]["station_name"]).values,
                             "y": va["target_pax"].values, "lgb": np.maximum(0, b + pl), "ridge": np.maximum(0, b + pr), "blend": np.maximum(0, b + 0.7 * pl + 0.3 * pr)}))
res = pd.concat(out); pickle.dump(res, open(os.path.join(os.path.dirname(ROOT), "oof_alex_ridge.pkl"), "wb")); print("сохранено", len(res))
