"""Стекирование: метамодели поверх прогнозов базовых моделей (все прогнозы — «месяц в проверку»).
Метамодель для проверочного месяца M учится ТОЛЬКО на прогнозах остальных трёх месяцев, прогнозирует месяц M.
S0 — простое среднее; S1 — NNLS-веса; S2 — NNLS по режимам (будни/выходные × пик/не пик); S3 — LightGBM-поправка к среднему
(признаки: отклонения моделей от среднего, разброс, слот, класс дня, станция, отклонение последних 30 мин, норма); S4 — S3 + NNLS-среднее.
Вход: ../ens_all.pkl (из ensemble_all.py). Запуск из корня репозитория."""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "app")); UP = os.path.dirname(ROOT)
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.optimize import nnls
import engine as E

d = pickle.load(open(os.path.join(UP, "ens_all.pkl"), "rb")); P = d["P"]; NS = E.NS
w = E.World(); S = np.arange(E.T); F = w.feat_rows(w.X, S); y = w.X.reshape(-1)
month = np.repeat(w.month, NS); slot = F["slot"].values.astype(int); st = F["st"].values.astype(int); cls = F["cls"].values.astype(int); norm = F["norm4"].values
a2 = (E.take(w.X, S - 2) + E.take(w.X, S - 3)).reshape(-1); n2 = (E.take(w.NORM, S - 2) + E.take(w.NORM, S - 3)).reshape(-1); dev_now = np.clip(np.nan_to_num(a2 / np.clip(n2, 1, None) - 1), -0.6, 1.0)
hour = ((slot * 15 + 180) // 60) % 24; peak = np.isin(hour, [7, 8, 9, 16, 17, 18]); wk = np.isin(cls, [5, 6])
wape = lambda mk, p: np.abs(y[mk] - p[mk]).sum() / y[mk].sum() * 100
def run(names, mk, tag):
    A = np.stack([P[k] for k in names], 1); mean = A.mean(1); out = {}
    out["S0 простое среднее"] = mean
    s1 = np.full(len(y), np.nan); s2 = s1.copy(); s3 = s1.copy(); s4 = s1.copy()
    Z = np.column_stack([np.log1p(mean), A.std(1) / np.clip(mean, 1, None), slot, cls, st, dev_now, np.log1p(np.nan_to_num(norm)), (A / np.clip(mean[:, None], 1, None) - 1)])
    for m in (2, 5, 7, 9):
        tr, te = mk & (month != m), mk & (month == m)
        wv, _ = nnls(A[tr], y[tr]); s1[te] = A[te] @ wv
        for r in (wk & peak, wk & ~peak, ~wk & peak, ~wk & ~peak):
            trr, ter = tr & r, te & r
            if trr.sum() > 200 and ter.any(): wr, _ = nnls(A[trr], y[trr]); s2[ter] = A[ter] @ wr
        rel = y / np.clip(mean, 1, None) - 1
        mdl = lgb.train(dict(objective="regression_l1", learning_rate=0.05, num_leaves=15, min_data_in_leaf=400, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=6),
                        lgb.Dataset(Z[tr], rel[tr], weight=mean[tr], categorical_feature=[3, 4]), 250)
        s3[te] = mean[te] * (1 + mdl.predict(Z[te])); s4[te] = 0.5 * s3[te] + 0.5 * s1[te]
    out.update({"S1 NNLS-веса": s1, "S2 NNLS по режимам": s2, "S3 LightGBM-поправка": np.clip(s3, 0, None), "S4 S3 + NNLS": np.clip(s4, 0, None)})
    rows = []
    for k, p in out.items():
        rows.append((k, wape(mk, p), wape(mk & ~wk, p), wape(mk & wk, p), wape(mk & peak, p), *[wape(mk & (month == m), p) for m in (2, 5, 7, 9)]))
    df = pd.DataFrame(rows, columns=["метод", "все", "будни", "выходные", "пик", "фев", "май", "июль", "сен"]).set_index("метод")
    best = {k: wape(mk, P[k]) for k in names}; b = min(best, key=best.get)
    print(f"\n=== {tag}: {mk.sum():,} строк, {len(names)} моделей; лучшая одиночная {b} = {best[b]:.2f} ===\n" + df.round(2).to_string())
base = ["M1 LGBM-L1", "M2 LGBM-Poisson", "M3 LGBM-остаток", "M4 CatBoost", "M5 инерция", "A Александр", "R Ridge", "N MLP", "N2 MLP-остаток"]
ok = d["ok"]; run(base, ok, "без Chronos-2")
okc = ok & ~np.isnan(P["CU Chronos-2"]) & ~np.isnan(P["CC Chronos-2 + норма"]); run(base + ["CU Chronos-2", "CC Chronos-2 + норма"], okc, "с Chronos-2")
