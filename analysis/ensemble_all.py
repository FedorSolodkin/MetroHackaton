"""Итоговое сравнение: наши бустинги (M1–M4), инерция (M5), норма (M6), модель Александра (A), Ridge (R), MLP (N), Chronos-2 (CU, CC)
и ансамбли. Все прогнозы — «месяц в проверку» (Chronos-2 без обучения). Сравнение на общих строках.
Ансамбль «жадный»: состав подбирается по трём месяцам (с повторами, как у Caruana et al.) и проверяется на четвёртом.
Запуск из корня репозитория после ensemble_lomo.py, oof_alexander.py, oof_nontree.py, oof_chronos2.py."""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "app")); UP = os.path.dirname(ROOT)
import numpy as np, pandas as pd
import engine as E

e = pickle.load(open(os.path.join(UP, "ens_oof.pkl"), "rb")); y, ev, month = e["y"], e["ev"], e["month"]; NS = E.NS
P = {k: v for k, v in e["oof"].items() if k.startswith("M")}
nt = pickle.load(open(os.path.join(UP, "oof_nontree.pkl"), "rb")); P.update(nt["oof"]); ok = ev & nt["have"]
a = pickle.load(open(os.path.join(UP, "oof_alexander.pkl"), "rb"))
gi = E.GRID.get_indexer(pd.to_datetime(a.target_datetime)); sj = a.station.map({s: i for i, s in enumerate(E.ST)}).values
idx = gi * NS + sj; pa = np.full(len(y), np.nan); good = (gi >= 0) & ~pd.isna(sj); pa[idx[good].astype(int)] = a.pred_alex.values[good]; P["A Александр"] = pa
ok &= ~np.isnan(pa)
for k in list(P): ok &= ~np.isnan(P[k])
has_c = os.path.exists(os.path.join(UP, "oof_chronos2.pkl"))
if has_c:
    c = pickle.load(open(os.path.join(UP, "oof_chronos2.pkl"), "rb")); kk = c["keys"][:, 0] * NS + c["keys"][:, 1]
    for nm, lab in (("chronos_u", "CU Chronos-2"), ("chronos_cov", "CC Chronos-2 + норма")):
        if nm in c: v = np.full(len(y), np.nan); v[kk] = c[nm]; P[lab] = v
    okc = ok.copy()
    for k in ("CU Chronos-2", "CC Chronos-2 + норма"):
        if k in P: okc &= ~np.isnan(P[k])
wape = lambda mk, p: np.abs(y[mk] - p[mk]).sum() / y[mk].sum() * 100
dow = np.repeat(pd.DatetimeIndex(E.GRID - pd.Timedelta(hours=3)).dayofweek.values, NS)
def greedy(names, mk, rounds=25):
    out = np.full(len(y), np.nan); comp = {}
    for m in (2, 5, 7, 9):
        tr, te = mk & (month != m), mk & (month == m); sel = []; cur = None
        for _ in range(rounds):
            best = min(names, key=lambda k: wape(tr, (cur * len(sel) + P[k]) / (len(sel) + 1) if sel else P[k]))
            sel.append(best); cur = np.mean([P[k] for k in sel], axis=0)
        out[te] = cur[te]; comp[m] = pd.Series(sel).value_counts().to_dict()
    return out, comp
def table(mk, names, title):
    print(f"\n=== {title}: {mk.sum():,} строк ===")
    rows = []
    for k in names:
        rows.append((k, wape(mk, P[k]), wape(mk & (dow < 5), P[k]), wape(mk & (dow >= 5), P[k]), *[wape(mk & (month == m), P[k]) for m in (2, 5, 7, 9)]))
    print(pd.DataFrame(rows, columns=["модель", "все", "пн–пт", "сб–вс", "фев", "май", "июль", "сен"]).set_index("модель").round(2).sort_values("все").to_string())
base = ["M1 LGBM-L1", "M2 LGBM-Poisson", "M3 LGBM-остаток", "M4 CatBoost", "M5 инерция", "A Александр", "R Ridge", "N MLP", "N2 MLP-остаток"]
P["E2 прежний (M1–M5)"] = np.mean([P[k] for k in base[:5]], axis=0)
P["E4 M1–M5 + Александр"] = np.mean([P[k] for k in base[:6]], axis=0)
P["E5 все 9"] = np.mean([P[k] for k in base], axis=0)
P["E7 M1–M5 + Александр + MLP-остаток"] = np.mean([P[k] for k in base[:6] + ["N2 MLP-остаток"]], axis=0)
P["G жадный (8 моделей)"], comp = greedy(base, ok)
table(ok, base + ["M6 норма", "E2 прежний (M1–M5)", "E4 M1–M5 + Александр", "E5 все 9", "E7 M1–M5 + Александр + MLP-остаток", "G жадный (8 моделей)"], "без Chronos (все строки)")
print("состав жадного ансамбля по проверочным месяцам:", comp)
if has_c and okc.any():
    cn = [k for k in ("CU Chronos-2", "CC Chronos-2 + норма") if k in P]
    P["E6 E4 + Chronos"] = np.mean([P[k] for k in base[:6] + cn], axis=0)
    P["G жадный (+Chronos)"], comp2 = greedy(base + cn, okc)
    table(okc, base + cn + ["M6 норма", "E4 M1–M5 + Александр", "E6 E4 + Chronos", "G жадный (8 моделей)", "G жадный (+Chronos)"], "с Chronos-2 (каждый второй слот)")
    print("состав жадного ансамбля (+Chronos):", comp2)
    errs = pd.DataFrame({k: (y - P[k])[okc] for k in base + cn}); print("\nкорреляция ошибок:\n", errs.corr().round(2).to_string())
pickle.dump({"P": P, "ok": ok}, open(os.path.join(UP, "ens_all.pkl"), "wb"))
