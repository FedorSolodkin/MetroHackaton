"""Сравнение наборов признаков: календарь / +погода / +лаги (nowcast 30 мин).
Проверка leave-one-month-out на 15-минутных данных, метрика WAPE.
Вход: pax15.pkl (ts, station, pax), weather_hourly.json (Open-Meteo).
"""
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import requests

PAX = sys.argv[1] if len(sys.argv) > 1 else "pax15.pkl"
WX = sys.argv[2] if len(sys.argv) > 2 else "weather_hourly.json"

# --- календарь: isdayoff.ru (0 рабочий, 1 выходной/праздник, 2 сокращённый предпраздничный)
txt = requests.get("https://isdayoff.ru/api/getdata?year=2026&pre=1", timeout=30).text
cal = pd.DataFrame({"day": pd.date_range("2026-01-01", periods=len(txt)), "code": [int(c) for c in txt]})
cal["off"] = (cal.code == 1).astype(int); cal["pre"] = (cal.code == 2).astype(int)
cal["off_next"] = cal.off.shift(-1).fillna(0).astype(int); cal["off_prev"] = cal.off.shift(1).fillna(0).astype(int)
cal.to_csv("calendar_2026.csv", index=False)

a = pd.read_pickle(PAX)
a = a[a.ts < "2026-10-01 03:00"]
a["day"] = (a.ts - pd.Timedelta(hours=3)).dt.normalize()
a["slot"] = ((a.ts - a.day - pd.Timedelta(hours=3)) // pd.Timedelta(minutes=15)).astype(int)
a["dow"] = a.day.dt.dayofweek; a["m"] = a.day.dt.month
a = a.merge(cal[["day", "off", "pre", "off_next", "off_prev"]], on="day", how="left")
a["st"] = a.station.astype("category").cat.codes

w = pd.DataFrame(json.load(open(WX, encoding="utf-8"))["hourly"]); w["time"] = pd.to_datetime(w.time)
w["rain3"] = w.precipitation.rolling(3, min_periods=1).sum(); w["rain6"] = w.precipitation.rolling(6, min_periods=1).sum()
w["dtemp3"] = w.temperature_2m.diff(3)
wcols = ["temperature_2m", "apparent_temperature", "precipitation", "rain3", "rain6", "snowfall", "snow_depth",
         "cloud_cover", "wind_gusts_10m", "relative_humidity_2m", "dtemp3", "weather_code"]
a["hour"] = a.ts.dt.floor("h"); a = a.merge(w[["time"] + wcols], left_on="hour", right_on="time", how="left")

# --- лаги на полной сетке 15 минут (разрывы между месяцами остаются NaN)
grid = pd.date_range("2026-02-01 03:00", "2026-10-01 02:45", freq="15min")
P = a.pivot(index="ts", columns="station", values="pax").reindex(grid)
lagcols = []
for k in (2, 3, 4, 6, 8, 96, 96 * 7):          # от 30 минут назад до недели назад
    L = P.shift(k).stack(future_stack=True).rename(f"lag{k}").reset_index(); L.columns = ["ts", "station", f"lag{k}"]
    a = a.merge(L, on=["ts", "station"], how="left"); lagcols.append(f"lag{k}")
a["mean_2_5"] = a[["lag2", "lag3", "lag4"]].mean(axis=1)
a["trend"] = a.lag2 - a.lag4
lagcols += ["mean_2_5", "trend"]

cal_f = ["st", "slot", "dow", "off", "pre", "off_next", "off_prev"]
sets = {"A календарь": cal_f, "B +погода": cal_f + wcols, "C +лаги 30мин": cal_f + wcols + lagcols}
wape = lambda y, p: np.abs(y - p).sum() / y.sum()
params = dict(objective="regression_l1", learning_rate=0.08, num_leaves=63, min_data_in_leaf=60,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
res = {k: {} for k in sets}; imp = None
for m in (2, 5, 7, 9):
    tr, te = a[a.m != m], a[a.m == m]
    for name, f in sets.items():
        mdl = lgb.train(params, lgb.Dataset(tr[f], tr.pax, categorical_feature=["st"]), num_boost_round=300)
        res[name][m] = wape(te.pax.values, mdl.predict(te[f]))
        if name.startswith("C") and m == 7:
            imp = pd.Series(mdl.feature_importance("gain"), index=f).sort_values(ascending=False)
out = pd.DataFrame(res).T; out["среднее"] = out.mean(axis=1)
print((out * 100).round(1).to_string())
print("\nтоп признаков (gain, модель C, проверка на июле):"); print((imp / imp.sum() * 100).round(1).head(12).to_string())
a.to_pickle("model_table.pkl")
