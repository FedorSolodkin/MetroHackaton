"""Ансамбль моделей, проверка «месяц в проверку» на уровне станций (19), все дни, слоты 06:30–00:15, горизонт 30 мин.
Модели: M1 LightGBM-L1 (как в приложении), M2 LightGBM-Poisson, M3 LightGBM на остатке (y - норма), M4 CatBoost-MAE,
M5 инерция (норма × (1 + 0.7·отклонение последних 30 мин)), M6 норма (медиана до 4 пред. дней того же класса).
Ансамбли: простое среднее и веса, подобранные на остальных месяцах (для проверочного месяца M веса учатся на прогнозах других месяцев).
Запуск из корня репозитория: python analysis/ensemble_lomo.py  -> печать таблицы, прогнозы сохраняются в ../ens_oof.pkl"""
import os, sys, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
import numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import nnls
import engine as E

t0 = time.time(); w = E.World(); S = np.arange(E.T)
F = w.feat_rows(w.X, S); y = w.X.reshape(-1); NS = E.NS
slot = np.repeat(w.slot, NS); month = np.repeat(w.month, NS); dow = np.repeat(w.dow, NS); off = np.repeat(w.off, NS); pre = np.repeat(w.pre, NS)
have = (slot >= 14) & (slot <= 85) & ~np.isnan(y) & np.isin(month, (2, 5, 7, 9))
norm = F["norm4"].values
# инерция для цели s: факт двух последних известных слотов (s-2, s-3) против нормы в них
Xn = w.NORM
a_now = (E.take(w.X, S - 2) + E.take(w.X, S - 3)).reshape(-1); n_now = (E.take(Xn, S - 2) + E.take(Xn, S - 3)).reshape(-1)
dev_now = np.clip(a_now / np.clip(n_now, 1, None) - 1, -0.6, 1.0); pers = norm * (1 + 0.7 * np.nan_to_num(dev_now))
P = dict(learning_rate=0.08, num_leaves=63, min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
oof = {k: np.full(len(y), np.nan) for k in ("M1 LGBM-L1", "M2 LGBM-Poisson", "M3 LGBM-остаток", "M4 CatBoost")}
Fc = F.copy(); Fc["st"] = Fc["st"].astype(int)
for m in (2, 5, 7, 9):
    tr = have & (month != m) & ~np.isnan(norm); te = have & (month == m)
    oof["M1 LGBM-L1"][te] = lgb.train({**P, "objective": "regression_l1"}, lgb.Dataset(F[tr], y[tr], categorical_feature=["st"]), 300).predict(F[te])
    oof["M2 LGBM-Poisson"][te] = lgb.train({**P, "objective": "poisson"}, lgb.Dataset(F[tr], y[tr], categorical_feature=["st"]), 400).predict(F[te])
    oof["M3 LGBM-остаток"][te] = np.nan_to_num(norm[te]) + lgb.train({**P, "objective": "regression_l1"}, lgb.Dataset(F[tr], y[tr] - norm[tr], categorical_feature=["st"]), 300).predict(F[te])
    cb = CatBoostRegressor(loss_function="MAE", iterations=600, learning_rate=0.1, depth=8, verbose=0, thread_count=4, cat_features=["st"])
    cb.fit(Fc[tr], y[tr]); oof["M4 CatBoost"][te] = cb.predict(Fc[te])
    print(f"месяц {m} готов, {time.time()-t0:.0f} с", flush=True)
oof["M5 инерция"] = pers; oof["M6 норма"] = norm
for k in oof: oof[k] = np.clip(oof[k], 0, None)
ev = have & ~np.isnan(norm) & ~np.isnan(pers)
base = ["M1 LGBM-L1", "M2 LGBM-Poisson", "M3 LGBM-остаток", "M4 CatBoost", "M5 инерция"]
oof["E1 среднее M1–M4"] = np.mean([oof[k] for k in base[:4]], axis=0)
oof["E2 среднее M1–M5"] = np.mean([oof[k] for k in base], axis=0)
# E3: неотрицательные веса, подобранные на других месяцах
e3 = np.full(len(y), np.nan); wts = {}
for m in (2, 5, 7, 9):
    trm = ev & (month != m); tem = ev & (month == m)
    A = np.stack([oof[k][trm] for k in base], 1); wv, _ = nnls(A, y[trm]); wts[m] = wv
    e3[tem] = np.stack([oof[k][tem] for k in base], 1) @ wv
oof["E3 веса по другим месяцам"] = e3
wape = lambda mk, k: np.abs(y[mk] - oof[k][mk]).sum() / y[mk].sum() * 100
grp = {"все дни": ev, "рабочие": ev & (dow < 5) & (off == 0) & (pre == 0), "выходные/праздн.": ev & ~((dow < 5) & (off == 0) & (pre == 0)),
       "пик 7–9,16–18": ev & np.isin(((slot * 15 + 180) // 60) % 24, [7, 8, 9, 16, 17, 18]), "аномалии |y−норма|>120": ev & (np.abs(y - np.nan_to_num(norm)) > 120)}
print("\nWAPE, % (месяц в проверку, станции × 15 мин)")
print(f"{'модель':30s}" + "".join(f"{g:>22s}" for g in grp) + "".join(f"{'мес '+str(m):>9s}" for m in (2, 5, 7, 9)))
for k in oof:
    print(f"{k:30s}" + "".join(f"{wape(mk, k):22.2f}" for mk in grp.values()) + "".join(f"{wape(ev & (month == m), k):9.2f}" for m in (2, 5, 7, 9)))
print("\nвеса E3 (M1..M5) по проверочным месяцам:", {m: np.round(v, 2).tolist() for m, v in wts.items()})
# теоретический минимум при чисто пуассоновском шуме (если бы мы знали точное среднее)
lam = np.clip(oof["E3 веса по другим месяцам"][ev], 0.5, None); floor = np.sqrt(2 * lam / np.pi).sum() / y[ev].sum() * 100
print(f"нижняя граница ошибки при чисто случайном (пуассоновском) шуме: ≈{floor:.1f}%")
cors = pd.DataFrame({k: (y[ev] - oof[k][ev]) for k in base}).corr().round(2); print("\nкорреляция ошибок моделей:\n", cors.to_string())
pickle.dump({"oof": oof, "y": y, "ev": ev, "month": month}, open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ens_oof.pkl"), "wb"))
print(f"\nвсего {time.time()-t0:.0f} с")
