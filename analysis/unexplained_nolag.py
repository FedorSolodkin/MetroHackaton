"""Неучтённые всплески: остаток модели (календарь+лаги, LOMO) -> кластеры по станциям."""
import pandas as pd, numpy as np, lightgbm as lgb
a = pd.read_pickle("model_table.pkl")
f = ["st","slot","dow","off","pre","off_next","off_prev","lag2","lag3","lag4","lag6","lag8","mean_2_5","trend"]
P = dict(objective="regression_l1", learning_rate=0.08, num_leaves=63, min_data_in_leaf=60, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
parts = []
for m in (2, 5, 7, 9):
    tr, te = a[a.m != m], a[a.m == m]
    mdl = lgb.train(P, lgb.Dataset(tr[f], tr.pax, categorical_feature=["st"]), 300)
    parts.append(te[["ts","day","dow","slot","station","pax"]].assign(p=mdl.predict(te[f])))
r = pd.concat(parts); r["res"] = r.pax - r.p
r.to_pickle("residuals_nolag.pkl")
# порог по станции: 3 робастных сигмы остатка, не меньше 120 чел/слот
sig = r.groupby("station").res.transform(lambda s: 1.4826 * (s - s.median()).abs().median())
r["hot"] = (r.res > np.maximum(3 * sig, 120))
r = r.sort_values(["station", "ts"]); r["grp"] = (~r.hot | (r.groupby("station").hot.shift(1) != True)).cumsum()
cl = r[r.hot].groupby(["station", "grp"]).agg(start=("ts","min"), end=("ts","max"), slots=("ts","size"), excess=("res","sum"), peak=("res","max")).reset_index()
cl = cl[cl.slots >= 3]; cl["end"] = cl.end + pd.Timedelta(minutes=15)
cl["dow"] = cl.start.dt.dayofweek; cl["hour"] = cl.start.dt.hour
print("кластеров (>=3 слотов подряд выше нормы):", len(cl), "| по станциям:"); print(cl.groupby("station").agg(n=("excess","size"), people=("excess","sum")).sort_values("people", ascending=False).round(0).head(10).to_string())
print("\nтоп-15 по избытку пассажиров:")
print(cl.nlargest(15, "excess")[["station","start","end","excess","peak"]].round(0).to_string(index=False))
rec = cl.groupby(["station","dow","hour"]).agg(n=("excess","size"), mean_excess=("excess","mean")).reset_index().query("n>=3").sort_values("n", ascending=False)
print("\nповторяющиеся (станция, день недели, час), >=3 раза:"); print(rec.head(12).round(0).to_string(index=False))
cl.to_csv("unexplained_nolag.csv", index=False)
