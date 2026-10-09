import sys; sys.path.insert(0, "repo/events")
import pandas as pd, numpy as np, lightgbm as lgb
from stations import station_of
a = pd.read_pickle("model_table.pkl"); r = pd.read_csv("rasp_schedule.csv", parse_dates=["time"])
r["dow"] = pd.to_datetime(r.date).dt.dayofweek; r["cs"] = r.time.dt.hour * 4 + r.time.dt.minute // 15
# матрица [станция метро][событие][dow][слот суток]
cnt = {}
for (st, ev, dw, cs), n in r.groupby(["metro_station", "event", "dow", "cs"]).size().items():
    cnt.setdefault((st, ev), np.zeros((7, 96)))[dw, cs] = n
a["ms"] = a.station.map(station_of)
cs = (a.ts.dt.hour * 4 + a.ts.dt.minute // 15).values; dw = a.ts.dt.dayofweek.values
a["arr30"] = 0.0; a["dep30"] = 0.0; a["arr60"] = 0.0
for st in set(r.metro_station):
    m = (a.ms == st).values
    A, D = cnt[(st, "arrival")], cnt[(st, "departure")]
    idx = np.where(m)[0]
    a.loc[a.index[idx], "arr30"] = [A[dw[i], max(cs[i]-2, 0):cs[i]].sum() for i in idx]
    a.loc[a.index[idx], "arr60"] = [A[dw[i], max(cs[i]-4, 0):cs[i]].sum() for i in idx]
    a.loc[a.index[idx], "dep30"] = [D[dw[i], cs[i]:cs[i]+2].sum() for i in idx]
base = ["st","slot","dow","off","pre","off_next","off_prev","lag2","lag3","lag4","lag6","lag8","lag96","lag672","mean_2_5","trend"]
rf = ["arr30", "arr60", "dep30"]
P = dict(objective="regression_l1", learning_rate=0.08, num_leaves=63, min_data_in_leaf=60, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
wape = lambda y, p: np.abs(y - p).sum() / y.sum()
sub = a.ms.isin(set(r.metro_station)); res = {"без вокзалов": {}, "с вокзалами": {}}; bystn = {}
for m in (2, 5, 7, 9):
    tr, te = a[a.m != m], a[a.m == m]; te = te[te.ms.isin(set(r.metro_station))]
    for n, f in (("без вокзалов", base), ("с вокзалами", base + rf)):
        mdl = lgb.train(P, lgb.Dataset(tr[f], tr.pax, categorical_feature=["st"]), 300)
        p = mdl.predict(te[f]); res[n][m] = wape(te.pax.values, p)
        if n == "с вокзалами": te = te.assign(p=p); bystn[m] = te
        else: te = te.assign(p0=p); bystn.setdefault("b", []).append(te)
o = pd.DataFrame(res).T; o["среднее"] = o.mean(axis=1); print("WAPE на 4 станциях у вокзалов (%):"); print((o * 100).round(1).to_string())
# по станциям, пик "вечер пятницы/воскресенья" – где дачники
b = pd.concat(bystn["b"]); w = pd.concat([bystn[m] for m in (2, 5, 7, 9)]); w["p0"] = b.p0.values
for st, g in w.groupby("ms"):
    print(f"  {st}: без {wape(g.pax.values, g.p0.values)*100:.1f} -> с {wape(g.pax.values, g.p.values)*100:.1f}")
fr = w[(w.dow.isin([4, 6])) & (w.ts.dt.hour.between(15, 21))]
print("пт/вс 15-21ч: без", round(wape(fr.pax.values, fr.p0.values)*100, 1), "-> с", round(wape(fr.pax.values, fr.p.values)*100, 1))
