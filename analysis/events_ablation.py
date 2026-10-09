"""Мелкие и средние мероприятия (Timepad + Афиша малых площадок) как признак по станциям.
Суммируем одновременные события у каждой станции (радиус 1.5 км, затухание по расстоянию), строим ev_in/ev_out/ev_n и
проверяем абляцией поверх календаря+лагов (месяц в проверку) + анализом остатков. Запуск из C:\metro."""
import sys
sys.path.insert(0, "repo/events")
import numpy as np, pandas as pd, lightgbm as lgb
from station_load import build
from stations import station_of

# --- события -----------------------------------------------------------------------------------
t = pd.read_pickle("timepad_geo.pkl")
t = t.dropna(subset=["lat2"]).copy()
SIZE = {"Экскурсии и путешествия": 25, "Искусство и культура": 80, "Для детей": 50, "Выставки": 120, "Хобби и творчество": 20,
        "Психология и самопознание": 25, "Театры": 200, "Концерты": 200}
t["capacity"] = t.category.map(SIZE).fillna(40)
ev = pd.DataFrame({"source": "timepad", "title": t.title, "category": t.category.fillna("other"), "venue": t.venue,
                   "lat": t.lat2, "lon": t.lon2, "start": t.start, "end": t.end, "capacity": t.capacity})
pl = pd.read_csv("afisha_places_events.csv", parse_dates=["datetime"])
ps = pl[pl.venue.isin(["Театр им. Ленсовета", "ДК «Выборгский»", "Космонавт"])]
ev = pd.concat([ev, pd.DataFrame({"source": "afisha", "title": ps.title, "category": ps.category.fillna("other"), "venue": ps.venue,
                "lat": ps.lat, "lon": ps.lon, "start": ps.datetime, "end": ps.datetime + pd.Timedelta(hours=2.5),
                "capacity": ps.venue.map({"Театр им. Ленсовета": 600, "ДК «Выборгский»": 800, "Космонавт": 2300})})])
ev = ev[(ev.start >= "2026-02-01") & (ev.start < "2026-10-01") & ev.start.dt.month.isin([2, 5, 7, 9])]
ev = ev[ev.capacity <= 2500]                       # только малые и средние
print("событий в наших месяцах (малые и средние, с координатами):", len(ev))
feat = build(ev, "2026-02-01", "2026-09-30").rename(columns={"station": "ms"})
nv = {}
a = pd.read_pickle("model_table.pkl"); a["ms"] = a.station.map(station_of)
for s, g in a.groupby("ms").station.nunique().items(): nv[s] = g
f = feat.copy(); f["k"] = f.ms.map(nv).fillna(1)
for c in ("ev_in", "ev_out", "ev_cont", "ev_n"): f[c] = f[c] / f.k          # делим на число вестибюлей
f = f.sort_values(["ms", "ts"])
for c in ("ev_in", "ev_out"):                                                  # накопленные за час (скользящая сумма по 4 слотам)
    f[c + "_1h"] = f.groupby("ms")[c].transform(lambda s: s.rolling(4, min_periods=1).sum())
a = a.merge(f[["ts", "ms", "ev_in", "ev_out", "ev_cont", "ev_n", "ev_in_1h", "ev_out_1h"]], on=["ts", "ms"], how="left").fillna(
    {"ev_in": 0, "ev_out": 0, "ev_cont": 0, "ev_n": 0, "ev_in_1h": 0, "ev_out_1h": 0})
a["ev_total"] = a.ev_in + a.ev_out
print("слотов с ev_total>0:", int((a.ev_total > 0).sum()), "| >5 чел.:", int((a.ev_total > 5).sum()), "| >20 чел.:", int((a.ev_total > 20).sum()))

# --- абляция -------------------------------------------------------------------------------------
base = ["st", "slot", "dow", "off", "pre", "off_next", "off_prev", "lag2", "lag3", "lag4", "lag6", "lag8", "lag96", "lag672", "mean_2_5", "trend"]
E = ["ev_in", "ev_out", "ev_cont", "ev_n", "ev_in_1h", "ev_out_1h"]
P = dict(objective="regression_l1", learning_rate=0.08, num_leaves=63, min_data_in_leaf=60, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
wape = lambda y, q: np.abs(y - q).sum() / y.sum()
parts = []
for m in (2, 5, 7, 9):
    tr, te = a[a.m != m], a[a.m == m].copy()
    for n, fs in (("C0", base), ("C0+события", base + E)):
        mdl = lgb.train(P, lgb.Dataset(tr[fs], tr.pax, categorical_feature=["st"]), 300); te[n] = mdl.predict(te[fs])
    parts.append(te[["ts", "day", "station", "ms", "m", "pax", "C0", "C0+события", "ev_in", "ev_out", "ev_total", "ev_n"]])
te = pd.concat(parts)
print("\nWAPE, % по месяцам:"); print(pd.DataFrame({n: {m: wape(te[te.m == m].pax.values, te[te.m == m][n].values) * 100 for m in (2, 5, 7, 9)} for n in ("C0", "C0+события")}).T.round(2).to_string())
for lab, mk in {"слоты с ожидаемым притоком >5 чел.": te.ev_total > 5, ">20 чел.": te.ev_total > 20, "станции с событиями (любое время)": te.ms.isin(te[te.ev_total > 0].ms.unique())}.items():
    d = te[mk]; print(f"  {lab:40s} n={len(d):6d}: C0={wape(d.pax.values, d.C0.values) * 100:.2f}  +события={wape(d.pax.values, d['C0+события'].values) * 100:.2f}")
rng = np.random.default_rng(0); g0 = te.assign(e0=(te.pax - te.C0).abs(), e1=(te.pax - te["C0+события"]).abs()).groupby("day").agg(e0=("e0", "sum"), e1=("e1", "sum"), y=("pax", "sum"))
dd = [(s.e0.sum() - s.e1.sum()) / s.y.sum() * 100 for s in (g0.loc[rng.choice(g0.index, len(g0))] for _ in range(1000))]
print(f"выигрыш по всем слотам, п.п. WAPE: {np.mean(dd):+.4f} [{np.percentile(dd, 2.5):+.4f}, {np.percentile(dd, 97.5):+.4f}]")
# остатки C0 против ожидаемого притока
te["rel"] = (te.pax - te.C0) / te.C0.clip(lower=30); d = te[te.C0 > 100]
d["bin"] = pd.cut(d.ev_total, [-1, 0, 2, 5, 15, 1e6])
print("\nотносительный остаток C0 по ожидаемому притоку событий (чел./слот):"); print(d.groupby("bin", observed=True).rel.agg(["mean", "median", "count"]).round(3).to_string())
