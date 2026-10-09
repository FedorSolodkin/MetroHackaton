"""Абляция: индекс пробки (скорость наземного транспорта) как признак поверх календаря+лагов. Месяц в проверку, парный бутстреп по дням.
Признаки берутся с запаздыванием >=30 мин (доступны на момент прогноза за 30 мин): d_speed станции, средний по линии,
доля станций с замедлением, доля стоящих. Вход: model_table.pkl, _gtfs/speed_*.csv. Запуск из C:\metro."""
import glob, sys
sys.path.insert(0, "repo/events")
import numpy as np, pandas as pd, lightgbm as lgb
from stations import station_of

sp = pd.concat([pd.read_csv(f, parse_dates=["ts"]) for f in sorted(glob.glob("_gtfs/speed_*.csv"))])
sp["day"] = (sp.ts - pd.Timedelta(hours=3)).dt.normalize(); sp["cls"] = np.where(sp.day.dt.dayofweek >= 5, "sat", "wd")
sp["slot"] = (((sp.ts.dt.hour * 60 + sp.ts.dt.minute - 180) % 1440) // 15).astype("int64"); sp = sp[sp.n_veh >= 3]
parts = []
for d in sp.day.unique():
    g = sp[sp.day != d].groupby(["station", "cls", "slot"]).agg(rs=("med_move", "median"), rp=("share_stopped", "median")).reset_index()
    parts.append(sp[sp.day == d].merge(g, on=["station", "cls", "slot"], how="left"))
sp = pd.concat(parts).dropna(subset=["rs"])
sp["d_speed"] = (sp.med_move - sp.rs) / sp.rs.clip(lower=1); sp["d_stop"] = sp.share_stopped - sp.rp
city = sp.groupby("ts").agg(city_d=("d_speed", "mean"), city_stop=("d_stop", "mean"), slow_share=("d_speed", lambda s: (s < -0.15).mean())).reset_index()
# сглаженный индекс: среднее за последний час (4 слота)
city = city.sort_values("ts"); city["city_d_1h"] = city.city_d.rolling(4, min_periods=2).mean()

def shifted(df, cols, k, suffix):
    o = df[["ts"] + (["station"] if "station" in df else []) + cols].copy(); o["ts"] = o.ts + pd.Timedelta(minutes=15 * k)
    return o.rename(columns={c: f"{c}_{suffix}" for c in cols})

a = pd.read_pickle("model_table.pkl"); a["ms"] = a.station.map(station_of)
spm = sp.rename(columns={"station": "ms"})[["ts", "ms", "d_speed", "d_stop"]]
for k, suf in ((2, "l2"), (3, "l3")):
    s = spm.copy(); s["ts"] = s.ts + pd.Timedelta(minutes=15 * k); s = s.rename(columns={"d_speed": f"d_speed_{suf}", "d_stop": f"d_stop_{suf}"})
    a = a.merge(s, on=["ts", "ms"], how="left")
for k, suf in ((2, "l2"), (3, "l3")):
    c = city.copy(); c["ts"] = c.ts + pd.Timedelta(minutes=15 * k)
    a = a.merge(c[["ts", "city_d", "city_stop", "slow_share", "city_d_1h"]].rename(columns={x: f"{x}_{suf}" for x in ("city_d", "city_stop", "slow_share", "city_d_1h")}), on="ts", how="left")
T = ["d_speed_l2", "d_speed_l3", "d_stop_l2", "city_d_l2", "city_d_l3", "city_stop_l2", "slow_share_l2", "city_d_1h_l2"]
print("слотов с индексом пробки:", int(a.city_d_l2.notna().sum()), "из", len(a))
base = ["st", "slot", "dow", "off", "pre", "off_next", "off_prev", "lag2", "lag3", "lag4", "lag6", "lag8", "lag96", "lag672", "mean_2_5", "trend"]
P = dict(objective="regression_l1", learning_rate=0.08, num_leaves=63, min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
wape = lambda y, q: np.abs(y - q).sum() / y.sum(); out = []
for m in (2, 5, 7, 9):
    tr, te = a[a.m != m], a[a.m == m].copy()
    for n, fs in (("C0", base), ("C0+пробки", base + T)):
        mdl = lgb.train(P, lgb.Dataset(tr[fs], tr.pax, categorical_feature=["st"]), 300); te[n] = mdl.predict(te[fs])
        if n == "C0+пробки" and m == 7: imp = pd.Series(mdl.feature_importance("gain"), index=fs).sort_values(ascending=False).head(8)
    out.append(te[["ts", "day", "m", "pax", "C0", "C0+пробки", "city_d_l2", "slow_share_l2"]])
te = pd.concat(out); h = te.ts.dt.hour; peak = h.between(7, 10) | h.between(16, 19)
print("\nWAPE, % по месяцам:"); print(pd.DataFrame({n: {m: wape(te[te.m == m].pax.values, te[te.m == m][n].values) * 100 for m in (2, 5, 7, 9)} for n in ("C0", "C0+пробки")}).T.round(2).to_string())
for lab, mk in {"пиковые часы": peak, "замедление по городу (city_d_l2 < -0.05)": te.city_d_l2 < -0.05, "замедление >=1 станции на 15%+ у половины станций": te.slow_share_l2 > 0.4, "ускорение (city_d_l2 > +0.05)": te.city_d_l2 > 0.05}.items():
    d = te[mk]; print(f"  {lab:50s} n={len(d):6d}: C0={wape(d.pax.values, d.C0.values) * 100:.2f}  +пробки={wape(d.pax.values, d['C0+пробки'].values) * 100:.2f}")
g = te.assign(e0=(te.pax - te.C0).abs(), e1=(te.pax - te["C0+пробки"]).abs()).groupby("day").agg(e0=("e0", "sum"), e1=("e1", "sum"), y=("pax", "sum"))
rng = np.random.default_rng(0); dd = [(s.e0.sum() - s.e1.sum()) / s.y.sum() * 100 for s in (g.loc[rng.choice(g.index, len(g))] for _ in range(1000))]
print(f"\nвыигрыш по всем слотам, п.п. WAPE: {np.mean(dd):+.3f} [95%: {np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}]  (плюс = лучше)")
print("\nважность признаков (июль в проверке):"); print((imp / imp.sum() * 100).round(1).to_string())
