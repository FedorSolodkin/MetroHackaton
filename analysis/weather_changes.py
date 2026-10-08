"""Погода как ИЗМЕНЕНИЕ: резкое похолодание/потепление, начало дождя после сухого периода, интенсивность осадков.
Абляция поверх календаря+лагов (30 мин назад, вчера, неделя), проверка 'месяц в проверку', парный бутстреп по дням.
Вход: model_table.pkl (из model_compare.py), weather_hourly.json. Запуск из C:\metro."""
import json
import numpy as np, pandas as pd, lightgbm as lgb

a = pd.read_pickle("model_table.pkl")
w = pd.DataFrame(json.load(open("weather_hourly.json", encoding="utf-8"))["hourly"]); w["hour"] = pd.to_datetime(w.time)
w = w.sort_values("hour").reset_index(drop=True)
t = w.temperature_2m; p = w.precipitation
w["d_t3"] = t - t.shift(3); w["d_t24"] = t - t.shift(24)
w["anom7"] = t - t.rolling(24 * 7, min_periods=48).mean()                 # отклонение от недельной нормы (только прошлое)
w["cold_snap"] = (w.anom7 < -4).astype(int); w["warm_spike"] = (w.anom7 > 4).astype(int)
w["rain_now"] = (p > 0.1).astype(int); r3 = p.rolling(3, min_periods=1).sum()
dry = (p.rolling(6, min_periods=1).sum().shift(1) < 0.1)                  # предыдущие 6 часов были сухими
w["rain_onset"] = ((p > 0.1) & dry).astype(int)
w["rain_onset_hard"] = ((p > 0.5) & dry).astype(int)
# часов с последних осадков
last = w.hour.where(p > 0.1).ffill(); w["dry_hours"] = ((w.hour - last).dt.total_seconds() / 3600).fillna(240).clip(upper=240)
w["precip_bin"] = pd.cut(p, [-1, 0.1, 0.5, 2, 100], labels=False)
w["rain3"] = r3; w["snow_now"] = (w.snowfall > 0.05).astype(int)
A = ["d_t3", "d_t24", "anom7", "cold_snap", "warm_spike"]; B = ["rain_now", "rain_onset", "rain_onset_hard", "dry_hours", "precip_bin", "rain3", "snow_now"]
drop = [c for c in A + B if c in a.columns]; a = a.drop(columns=drop)
a = a.merge(w[["hour"] + A + B], on="hour", how="left")

base = ["st", "slot", "dow", "off", "pre", "off_next", "off_prev", "lag2", "lag3", "lag4", "lag6", "lag8", "lag96", "lag672", "mean_2_5", "trend"]
sets = {"C0 календарь+лаги": base, "+A изменения температуры": base + A, "+B осадки и их начало": base + B, "+A+B": base + A + B}
P = dict(objective="regression_l1", learning_rate=0.08, num_leaves=63, min_data_in_leaf=60, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
wape = lambda y, q: np.abs(y - q).sum() / y.sum()
preds = {k: [] for k in sets}; keep = []
for m in (2, 5, 7, 9):
    tr, te = a[a.m != m], a[a.m == m]
    keep.append(te[["ts", "day", "station", "pax", "slot", "m", "d_t24", "rain_onset_hard", "cold_snap", "warm_spike", "precip_bin"]])
    for n, f in sets.items():
        mdl = lgb.train(P, lgb.Dataset(tr[f], tr.pax, categorical_feature=["st"]), 300)
        preds[n].append(mdl.predict(te[f]))
te = pd.concat(keep).reset_index(drop=True)
for n in sets: te[n] = np.concatenate(preds[n])
print("WAPE, % (месяц в проверку):")
rows = {n: {m: wape(te[te.m == m].pax.values, te[te.m == m][n].values) * 100 for m in (2, 5, 7, 9)} for n in sets}
o = pd.DataFrame(rows).T; o["среднее"] = o.mean(axis=1); print(o.round(2).rename(columns={2: "фев", 5: "май", 7: "июль", 9: "сен"}).to_string())
# где погода могла сработать: пик и "события погоды"
hr = te.ts.dt.hour; peak = hr.between(7, 10) | hr.between(16, 19)
sub = {"пиковые часы": peak, "резкое изменение t за сутки (|d_t24|>4)": te.d_t24.abs() > 4, "начало сильного дождя": te.rain_onset_hard == 1,
       "похолодание/потепление >4° к неделе": (te.cold_snap == 1) | (te.warm_spike == 1), "осадки >0.5 мм/ч": te.precip_bin >= 2}
print("\nWAPE на подвыборках (n слотов):")
for k, mk in sub.items():
    d = te[mk]; print(f"  {k:42s} n={len(d):6d} | " + " | ".join(f"{n[:6]}={wape(d.pax.values, d[n].values) * 100:5.2f}" for n in sets))
# парный бутстреп по дням: C0 против +A+B
days = te.day.unique(); rng = np.random.default_rng(0)
e0 = te.assign(e=(te.pax - te["C0 календарь+лаги"]).abs()).groupby("day").agg(e0=("e", "sum"), y=("pax", "sum"))
e1 = te.assign(e=(te.pax - te["+A+B"]).abs()).groupby("day").e.sum(); e0["e1"] = e1
d = []
for _ in range(1000):
    s = e0.loc[rng.choice(e0.index, len(e0))]; d.append((s.e0.sum() - s.e1.sum()) / s.y.sum() * 100)
print(f"\nвыигрыш +A+B против C0 (п.п. WAPE), 95% интервал по дням: {np.mean(d):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
