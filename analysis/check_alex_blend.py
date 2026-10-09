"""Проверка бленда Александра (0.7·LightGBM-P50 + 0.3·Ridge на остатках) «месяц в проверку», горизонты 15/30/60/120, его признаки и параметры."""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import Ridge
from src.data.data_loader import MetroDataLoader
from src.models.baseline import MetroBaselineModel
from src.models.feature_builder import MetroFeatureBuilder, CATEGORICAL_FEATURES
from src.models.train_multi_horizon import MultiHorizonFlowModel
L = MetroDataLoader(); dfs = L.get_station_aggregated_flow(L.load_15min_dataset()); dfs["month"] = pd.to_datetime(dfs["datetime"]).dt.month
wape = lambda y, p: np.abs(y - p).sum() / y.sum() * 100; t0 = time.time(); mh = MultiHorizonFlowModel(); rows = []
for H in (15, 30, 60, 120):
    for m in (2, 5, 7, 9):
        tr_raw, va_raw = dfs[dfs.month != m].copy(), dfs[dfs.month == m].copy(); fb = MetroBaselineModel(); fb.fit(tr_raw); bld = MetroFeatureBuilder(baseline_model=fb)
        tr, fc = bld.build_training_dataset_for_horizon(tr_raw, horizon_min=H); va, _ = bld.build_training_dataset_for_horizon(va_raw, horizon_min=H)
        bst = lgb.train(mh._get_lgb_params(0.5, H), lgb.Dataset(tr[fc], label=tr["target_res"], categorical_feature=CATEGORICAL_FEATURES), num_boost_round=120)
        rg = Ridge(alpha=100.0, random_state=42).fit(tr[fc].fillna(0.0), tr["target_res"])
        b = va["base_pax_target"].values; y = va["target_pax"].values; pl = bst.predict(va[fc]); pr = rg.predict(va[fc].fillna(0.0)); pb = 0.7 * pl + 0.3 * pr
        rows.append((H, m, wape(y, b), wape(y, np.maximum(0, b + pl)), wape(y, np.maximum(0, b + pr)), wape(y, np.maximum(0, b + pb))))
    print(f"H={H} готов, {time.time()-t0:.0f} с", flush=True)
df = pd.DataFrame(rows, columns=["H", "месяц", "база", "LightGBM", "Ridge", "бленд 70/30"])
print("\nWAPE, % (среднее по 4 месяцам):"); print(df.groupby("H")[["база", "LightGBM", "Ridge", "бленд 70/30"]].mean().round(2).to_string())
print("\nпо месяцам, H=30:"); print(df[df.H == 30].set_index("месяц")[["база", "LightGBM", "Ridge", "бленд 70/30"]].round(2).to_string())
