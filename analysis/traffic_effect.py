"""Связь пробок (скорость наземного транспорта вокруг станции) с отклонением потока метро от прогноза.
Индекс: d_speed = (скорость - норма)/норма, норма = медиана по остальным дням того же типа (будни/суббота) для станции и слота.
Проверка: корреляция d_speed(t-k) с относительным остатком модели 'календарь+лаги' в слоте t (k=0..4 слота = 0..60 мин вперёд).
Вход: _gtfs/speed_*.csv, residuals.pkl. Запуск из C:\metro."""
import glob, sys
sys.path.insert(0, "repo/events")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from stations import station_of

sp = pd.concat([pd.read_csv(f, parse_dates=["ts"]) for f in sorted(glob.glob("_gtfs/speed_*.csv"))])
sp["day"] = (sp.ts - pd.Timedelta(hours=3)).dt.normalize(); sp["dow"] = sp.day.dt.dayofweek; sp["cls"] = np.where(sp.dow >= 5, "sat", "wd")
sp["slot"] = (((sp.ts.dt.hour * 60 + sp.ts.dt.minute - 180) % 1440) // 15).astype("int64")
sp = sp[sp.n_veh >= 3]                                                   # достаточно машин в радиусе
ref = []
for d in sp.day.unique():                                                # норма без текущего дня
    o = sp[(sp.day != d)]; cur = sp[sp.day == d]
    g = o.groupby(["station", "cls", "slot"]).agg(ref_speed=("med_move", "median"), ref_stop=("share_stopped", "median")).reset_index()
    ref.append(cur.merge(g, on=["station", "cls", "slot"], how="left"))
sp = pd.concat(ref).dropna(subset=["ref_speed"])
sp["d_speed"] = (sp.med_move - sp.ref_speed) / sp.ref_speed.clip(lower=1); sp["d_stop"] = sp.share_stopped - sp.ref_stop
print("дней:", sp.day.nunique(), "| станций:", sp.station.nunique(), "| слот-станций:", len(sp))
print("d_speed: P5/P50/P95 =", sp.d_speed.quantile([.05, .5, .95]).round(2).tolist(), "| доля замедлений >25%:", round((sp.d_speed < -0.25).mean(), 3))

r = pd.read_pickle("residuals.pkl"); r["ms"] = r.station.map(station_of); r["rel"] = (r.pax - r.p) / r.p.clip(lower=30); r = r[r.p > 100]
r = r.groupby(["ts", "ms"]).rel.mean().reset_index()                     # станция целиком (вестибюли усредняем)
res = {}
for k in range(0, 5):
    s = sp[["ts", "station", "d_speed", "d_stop"]].copy(); s["ts"] = s.ts + pd.Timedelta(minutes=15 * k)    # скорость k слотов назад -> слот t
    m = r.merge(s, left_on=["ts", "ms"], right_on=["ts", "station"]).dropna(subset=["d_speed"])
    res[k] = (spearmanr(m.d_speed, m.rel)[0], spearmanr(m.d_stop, m.rel)[0], len(m))
print("\nкорреляция Спирмена: остаток метро в слоте t и пробка k слотов назад")
for k, (a, b, n) in res.items(): print(f"  опережение {15*k:2d} мин: скорость {a:+.3f} | доля стоящих {b:+.3f} (n={n})")
m = r.merge(sp[["ts", "station", "d_speed"]], left_on=["ts", "ms"], right_on=["ts", "station"])
m["hr"] = m.ts.dt.hour
m["bin"] = pd.cut(m.d_speed, [-2, -0.3, -0.15, 0.15, 0.3, 5])
print("\nотносительный остаток метро по замедлению транспорта (скорость к норме, в слоте t):"); print(m.groupby("bin", observed=True).rel.agg(["mean", "median", "count"]).round(3).to_string())
pk = m[m.hr.isin([7, 8, 9, 16, 17, 18, 19])]
print("\nтолько пиковые часы:"); print(pk.groupby("bin", observed=True).rel.agg(["mean", "median", "count"]).round(3).to_string())
# по линии: усреднённая по станциям пробка и средний остаток по линии
L = sp.groupby("ts").d_speed.mean().rename("city_d"); Rm = r.groupby("ts").rel.mean().rename("metro_rel"); J = pd.concat([L, Rm], axis=1).dropna()
print("\nпо линии целиком (среднее по станциям), корреляция Спирмена по слотам:", round(spearmanr(J.city_d, J.metro_rel)[0], 3), "n=", len(J))
for k in (1, 2, 3):
    J2 = pd.concat([L.shift(k, freq="15min"), Rm], axis=1).dropna(); print(f"   пробка за {15*k} мин до: {spearmanr(J2.iloc[:,0], J2.iloc[:,1])[0]:+.3f}")
