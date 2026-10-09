"""Предсказуемость всплесков: помогают ли погода (изменения) и мелкие события предсказать, что поток в слоте сильно выше
прогноза (остаток > max(3 сигмы станции, 120 чел)). Метрики: ROC-AUC и средняя точность (PR-AUC), месяц в проверку.
Вход: residuals.pkl (остатки календарь+лаги), weather_hourly.json, timepad_geo.pkl, afisha_places_events.csv. Запуск из C:\metro."""
import json, sys
sys.path.insert(0, "repo/events")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from station_load import build
from stations import station_of

r = pd.read_pickle("residuals.pkl"); r["ms"] = r.station.map(station_of); r["hour"] = r.ts.dt.floor("h"); r["m"] = r.ts.dt.month
sig = r.groupby("station").res.transform(lambda s: 1.4826 * (s - s.median()).abs().median())
r["hot"] = (r.res > np.maximum(3 * sig, 120)).astype(int)
w = pd.DataFrame(json.load(open("weather_hourly.json", encoding="utf-8"))["hourly"]); w["hour"] = pd.to_datetime(w.time); w = w.sort_values("hour").reset_index(drop=True)
t = w.temperature_2m; p = w.precipitation
w["temp"] = t; w["d_t24"] = t - t.shift(24); w["anom7"] = t - t.rolling(168, min_periods=48).mean(); w["precip"] = p
w["onset"] = ((p > 0.3) & (p.rolling(6, min_periods=1).sum().shift(1) < 0.1)).astype(int); w["dry_h"] = ((w.hour - w.hour.where(p > 0.1).ffill()).dt.total_seconds() / 3600).fillna(240).clip(upper=240)
r = r.merge(w[["hour", "temp", "d_t24", "anom7", "precip", "onset", "dry_h"]], on="hour", how="left")
# события малые и средние (как в events_ablation.py)
tp = pd.read_pickle("timepad_geo.pkl").dropna(subset=["lat2"]); SIZE = {"Экскурсии и путешествия": 25, "Искусство и культура": 80, "Для детей": 50, "Выставки": 120, "Хобби и творчество": 20, "Психология и самопознание": 25, "Театры": 200, "Концерты": 200}
ev = pd.DataFrame({"source": "t", "title": tp.title, "category": tp.category.fillna("other"), "venue": tp.venue, "lat": tp.lat2, "lon": tp.lon2, "start": tp.start, "end": tp.end, "capacity": tp.category.map(SIZE).fillna(40)})
ev = ev[(ev.start >= "2026-02-01") & (ev.start < "2026-10-01") & ev.start.dt.month.isin([2, 5, 7, 9])]
f = build(ev, "2026-02-01", "2026-09-30").rename(columns={"station": "ms"})
r = r.merge(f[["ts", "ms", "ev_in", "ev_out", "ev_cont", "ev_n"]], on=["ts", "ms"], how="left").fillna(0)
r["st"] = r.station.astype("category").cat.codes; r["slot"] = ((r.ts.dt.hour * 60 + r.ts.dt.minute - 180) % 1440) // 15; r["dow"] = r.ts.dt.dayofweek
r = r[r.p > 100]
print(f"слотов: {len(r)}, всплесков (hot): {int(r.hot.sum())} ({r.hot.mean():.2%})")
T = ["st", "slot", "dow"]; Wf = ["temp", "d_t24", "anom7", "precip", "onset", "dry_h"]; Ev = ["ev_in", "ev_out", "ev_cont", "ev_n"]
sets = {"время+станция": T, "+погода (изменения)": T + Wf, "+мелкие события": T + Ev, "+всё": T + Wf + Ev}
P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
res = {k: {"auc": [], "ap": []} for k in sets}
for m in (2, 5, 7, 9):
    tr, te = r[r.m != m], r[r.m == m]
    for n, fs in sets.items():
        mdl = lgb.train(P, lgb.Dataset(tr[fs], tr.hot, categorical_feature=["st"]), 250); pr = mdl.predict(te[fs])
        res[n]["auc"].append(roc_auc_score(te.hot, pr)); res[n]["ap"].append(average_precision_score(te.hot, pr))
print("\nROC-AUC (0.5 = случайно) и средняя точность, среднее по 4 месяцам; базовая доля всплесков = %.3f" % r.hot.mean())
for n in sets: print(f"  {n:24s} AUC={np.mean(res[n]['auc']):.3f}  AP={np.mean(res[n]['ap']):.3f}   по месяцам AUC: " + " ".join(f"{x:.3f}" for x in res[n]["auc"]))
