"""Стекирование и среднее с новыми моделями: GRU по линии (Q), аналоги (K), TimesFM (T, если посчитан). Вход: ../ens_all.pkl, ../oof_new.pkl, ../oof_timesfm.pkl."""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "app")); UP = os.path.dirname(ROOT)
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.optimize import nnls
import engine as E
d = pickle.load(open(os.path.join(UP, "ens_all.pkl"), "rb")); P = d["P"]; NS = E.NS; ok = d["ok"].copy()
P.update(pickle.load(open(os.path.join(UP, "oof_new.pkl"), "rb")))
if os.path.exists(os.path.join(UP, "oof_timesfm.pkl")):
    t = pickle.load(open(os.path.join(UP, "oof_timesfm.pkl"), "rb")); v = np.full(E.T * NS, np.nan); v[t["keys"][:, 0] * NS + t["keys"][:, 1]] = t["timesfm"]; P["T TimesFM"] = v
w = E.World(); S = np.arange(E.T); F = w.feat_rows(w.X, S); y = w.X.reshape(-1)
month = np.repeat(w.month, NS); slot = F["slot"].values.astype(int); st = F["st"].values.astype(int); cls = F["cls"].values.astype(int); norm = F["norm4"].values
a2 = (E.take(w.X, S - 2) + E.take(w.X, S - 3)).reshape(-1); n2 = (E.take(w.NORM, S - 2) + E.take(w.NORM, S - 3)).reshape(-1); dev_now = np.clip(np.nan_to_num(a2 / np.clip(n2, 1, None) - 1), -0.6, 1.0)
hour = ((slot * 15 + 180) // 60) % 24; peak = np.isin(hour, [7, 8, 9, 16, 17, 18]); wk = np.isin(cls, [5, 6])
wape = lambda mk, p: np.abs(y[mk] - p[mk]).sum() / y[mk].sum() * 100
old = ["M1 LGBM-L1", "M2 LGBM-Poisson", "M3 LGBM-остаток", "M4 CatBoost", "M5 инерция", "A Александр", "R Ridge", "N MLP", "N2 MLP-остаток"]; ch = ["CU Chronos-2", "CC Chronos-2 + норма"]
def stack(names, mk):
    A = np.stack([P[k] for k in names], 1); mean = A.mean(1); s3 = np.full(len(y), np.nan); s1 = s3.copy()
    Z = np.column_stack([np.log1p(mean), A.std(1) / np.clip(mean, 1, None), slot, cls, st, dev_now, np.log1p(np.nan_to_num(norm)), (A / np.clip(mean[:, None], 1, None) - 1)]); rel = y / np.clip(mean, 1, None) - 1
    for m in (2, 5, 7, 9):
        tr, te = mk & (month != m), mk & (month == m); wv, _ = nnls(A[tr], y[tr]); s1[te] = A[te] @ wv
        mdl = lgb.train(dict(objective="regression_l1", learning_rate=0.05, num_leaves=15, min_data_in_leaf=400, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=6),
                        lgb.Dataset(Z[tr], rel[tr], weight=mean[tr], categorical_feature=[3, 4]), 250); s3[te] = mean[te] * (1 + mdl.predict(Z[te]))
    return mean, s1, np.clip(s3, 0, None)
def report(names, mk, tag):
    mean, s1, s3 = stack(names, mk); rows = []
    for lab, p in (("простое среднее", mean), ("NNLS-веса", s1), ("стекер LightGBM", s3)):
        rows.append((lab, wape(mk, p), wape(mk & ~wk, p), wape(mk & wk, p), wape(mk & peak, p), *[wape(mk & (month == m), p) for m in (2, 5, 7, 9)]))
    print(f"\n=== {tag}: {mk.sum():,} строк, {len(names)} моделей ===\n" + pd.DataFrame(rows, columns=["метод", "все", "будни", "выходные", "пик", "фев", "май", "июль", "сен"]).set_index("метод").round(2).to_string())
new = ["Q GRU по линии", "K аналоги"]; mk_new = ok & ~np.isnan(P["Q GRU по линии"]) & ~np.isnan(P["K аналоги"])
print("поштучно на общих строках:", {k: round(wape(mk_new, P[k]), 2) for k in new + ["N2 MLP-остаток", "M1 LGBM-L1", "M6 норма"]})
report(old, mk_new, "ДО: прежние 9 моделей"); report(old + new, mk_new, "ПОСЛЕ: + GRU по линии + аналоги")
okc = mk_new & ~np.isnan(P["CU Chronos-2"]) & ~np.isnan(P["CC Chronos-2 + норма"])
report(old + ch, okc, "прежние 9 + Chronos-2"); report(old + ch + new, okc, "прежние 9 + Chronos-2 + GRU + аналоги")
if "T TimesFM" in P:
    okt = okc & ~np.isnan(P["T TimesFM"]); print("TimesFM поштучно:", round(wape(okt, P["T TimesFM"]), 2)); report(old + ch + new, okt, "(та же выборка) без TimesFM"); report(old + ch + new + ["T TimesFM"], okt, "+ TimesFM")
errs = pd.DataFrame({k: (y - P[k])[okc] for k in ["M1 LGBM-L1", "N2 MLP-остаток", "M5 инерция", "A Александр", "CC Chronos-2 + норма"] + new + (["T TimesFM"] if "T TimesFM" in P else [])}).dropna(); print("\nкорреляция ошибок:\n", errs.corr().round(2).to_string())
