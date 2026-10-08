"""Медиана-бейзлайн по (станция, тип дня, слот) на пассажирских слотах 05:30-00:30, проверка месяц-в-проверку.
Варианты: B0 - как раньше (все слоты, wd/sat/sun/hol); B1 - те же типы, только пассажирские слоты;
B2 - пассажирские слоты и типы Пн-Чт / Пт / Сб / Вс / праздник / предпраздничный (календарь isdayoff).
Вход: pax15.pkl (ts, station, pax), calendar_2026.csv. Запуск из C:\metro."""
import numpy as np, pandas as pd

a = pd.read_pickle("pax15.pkl"); a = a[a.ts < "2026-10-01 03:00"].copy()
cal = pd.read_csv("calendar_2026.csv", parse_dates=["day"])
a["day"] = (a.ts - pd.Timedelta(hours=3)).dt.normalize()
a["slot"] = ((a.ts - a.day - pd.Timedelta(hours=3)) // pd.Timedelta(minutes=15)).astype(int)
a["dow"] = a.day.dt.dayofweek; a["m"] = a.day.dt.month
mod = a.ts.dt.hour * 60 + a.ts.dt.minute
a["oper"] = (mod >= 330) | (mod < 30)
a = a.merge(cal[["day", "code"]], on="day", how="left")
hol_old = {"2026-02-23", "2026-05-01", "2026-05-09", "2026-05-11"}
a["t0"] = np.where(a.day.dt.strftime("%Y-%m-%d").isin(hol_old), "hol", np.where(a.dow < 5, "wd", np.where(a.dow == 5, "sat", "sun")))
off = (a.code == 1); pre = (a.code == 2)
a["t2"] = np.select([off & (a.dow < 5), pre, a.dow == 4, a.dow < 4, a.dow == 5], ["hol", "pre", "fri", "wd", "sat"], "sun")
fallback = {"hol": "sun", "pre": "fri"}   # если типа нет в обучающих месяцах

def wape(y, p): return np.abs(y - p).sum() / y.sum()

def run(df, col, test_month, fb=None):
    tr, te = df[df.m != test_month], df[df.m == test_month].copy()
    med = tr.groupby(["station", col, "slot"]).pax.median().rename("base").reset_index()
    t = te.merge(med, on=["station", col, "slot"], how="left")
    if fb:
        miss = t.base.isna()
        if miss.any():
            t2 = t.loc[miss].drop(columns="base").assign(**{col: t.loc[miss, col].map(lambda x: fb.get(x, x))})
            t.loc[miss, "base"] = t2.merge(med, on=["station", col, "slot"], how="left").base.values
    t = t.dropna(subset=["base"]); return wape(t.pax.values, t.base.values), len(t)

res = {}
for name, df, col, fb in (("B0 все слоты, wd/sat/sun/hol", a, "t0", None),
                          ("B1 пасс. слоты, wd/sat/sun/hol", a[a.oper], "t0", None),
                          ("B2 пасс. слоты, пн-чт/пт/сб/вс/hol/pre", a[a.oper], "t2", fallback)):
    res[name] = {m: run(df, col, m, fb)[0] for m in (2, 5, 7, 9)}
o = pd.DataFrame(res).T; o["среднее"] = o.mean(axis=1)
print("WAPE, % (медиана по станции×типу дня×слоту; проверка: месяц целиком не участвует в медиане)")
print((o * 100).round(1).rename(columns={2: "фев", 5: "май", 7: "июль", 9: "сен"}).to_string())
# где B2 выигрывает: пятницы и предпраздничные дни
b = a[a.oper]
for m in (2, 5, 7, 9):
    pass
fr = {}
for typ in ("fri", "pre", "hol"):
    d = b[b.t2 == typ]
    if len(d) == 0: continue
    errs = []
    for m in sorted(d.m.unique()):
        tr = b[b.m != m]; te = d[d.m == m]
        med0 = tr.groupby(["station", "t0", "slot"]).pax.median().rename("b0").reset_index()
        med2 = tr.groupby(["station", "t2", "slot"]).pax.median().rename("b2").reset_index()
        t = te.merge(med0, on=["station", "t0", "slot"], how="left").merge(med2, on=["station", "t2", "slot"], how="left").dropna(subset=["b0", "b2"])
        if len(t): errs.append((wape(t.pax.values, t.b0.values), wape(t.pax.values, t.b2.values), len(t)))
    if errs:
        w = np.array([e[2] for e in errs]); fr[typ] = (np.average([e[0] for e in errs], weights=w), np.average([e[1] for e in errs], weights=w))
print("\nна отдельных типах дней (B1 -> B2), WAPE %:", {k: (round(v[0] * 100, 1), round(v[1] * 100, 1)) for k, v in fr.items()})
print("число дней по типам:", b.groupby("t2").day.nunique().to_dict())
