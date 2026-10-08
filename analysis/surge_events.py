"""Неожиданные всплески потока на линии и доля, которую модель видит за 30 мин (прогноз LightGBM календарь+лаги).
Всплеск: поток по линии > нормы (медиана по станции×типу дня×слоту из остальных месяцев) на THR и более, >=MINRUN слотов подряд,
в пассажирские часы, кроме праздников и предпраздничных дней (их диспетчеры знают).
Обнаружен: прогноз модели в слоте поднимается над нормой минимум на половину фактического превышения.
Вход: pax15.pkl, residuals.pkl, calendar_2026.csv. Запуск из C:\metro."""
import numpy as np, pandas as pd
THR, MINRUN = 0.10, 2
a = pd.read_pickle("pax15.pkl"); a = a[a.ts < "2026-10-01 03:00"].copy()
cal = pd.read_csv("calendar_2026.csv", parse_dates=["day"])
a["day"] = (a.ts - pd.Timedelta(hours=3)).dt.normalize(); a["slot"] = (((a.ts.dt.hour * 60 + a.ts.dt.minute - 180) % 1440) // 15).astype("int64")
a["dow"] = a.day.dt.dayofweek; a["m"] = a.day.dt.month; a = a.merge(cal[["day", "code"]], on="day", how="left")
mod = a.ts.dt.hour * 60 + a.ts.dt.minute; a = a[(mod >= 330) | (mod < 30)]
a["t"] = np.select([(a.code == 1) & (a.dow < 5), a.code == 2, a.dow == 4, a.dow < 4, a.dow == 5], ["hol", "pre", "fri", "wd", "sat"], "sun")
KNOWN_HOLIDAYS = pd.to_datetime(["2026-02-21","2026-02-22","2026-02-23","2026-05-01","2026-05-02","2026-05-03","2026-05-09","2026-05-10","2026-05-11"])
a.loc[a.day.isin(KNOWN_HOLIDAYS), "t"] = "hol"
base = []
for m in (2, 5, 7, 9):
    g = a[a.m != m].groupby(["station", "t", "slot"]).pax.median().rename("base").reset_index()
    base.append(a[a.m == m].merge(g, on=["station", "t", "slot"], how="left"))
a = pd.concat(base).dropna(subset=["base"])
r = pd.read_pickle("residuals.pkl")[["ts", "station", "p"]]; a = a.merge(r, on=["ts", "station"], how="inner")
L = a.groupby(["ts", "day", "m", "t"]).agg(pax=("pax", "sum"), base=("base", "sum"), p=("p", "sum")).reset_index().sort_values("ts")
L["ex"] = L.pax / L.base - 1; L["surge"] = (L.ex > THR) & ~L.t.isin(["hol", "pre"])
L["run"] = (L.surge != L.surge.shift()) | (L.day != L.day.shift()); L["rid"] = L.run.cumsum()
ev = L[L.surge].groupby("rid").agg(day=("day", "first"), m=("m", "first"), start=("ts", "min"), slots=("ts", "size"), extra_pax=("pax", lambda s: 0))
for rid, g in L[L.surge].groupby("rid"):
    ev.loc[rid, "extra_pax"] = float((g.pax - g.base).sum()); ev.loc[rid, "detect"] = ((g.p - g.base) >= 0.5 * (g.pax - g.base)).mean(); ev.loc[rid, "peak_ex"] = g.ex.max()
ev = ev[ev.slots >= MINRUN]; ev["hr"] = ev.start.dt.hour
print(f"порог +{THR:.0%} к норме, >= {MINRUN} слотов подряд ({MINRUN*15} мин), без праздничных и предпраздничных дней")
print("всего событий:", len(ev), "| дней в выборке:", L.day.nunique(), "| событий в день: %.2f" % (len(ev) / L.day.nunique()))
print("по месяцам:", ev.groupby("m").size().to_dict(), "| на 30 дней: %.1f" % (len(ev) / L.day.nunique() * 30))
print("длительность, мин: P50/P90 =", (ev.slots.quantile([.5, .9]) * 15).tolist(), "| лишних пассажиров за событие P50/P90:", ev.extra_pax.quantile([.5, .9]).round(0).tolist())
print("доля слотов события, где модель (за 30 мин) показывает >=половины превышения: %.2f" % ev.detect.mean(), "| события, обнаруженные хотя бы в половине слотов: %.2f" % (ev.detect >= 0.5).mean())
big = ev[(ev.peak_ex >= 0.2)]; print(f"крупные (превышение не менее 20 процентов): {len(big)} ({len(big)/L.day.nunique()*30:.1f} на 30 дней), обнаружены {(big.detect >= 0.5).mean():.2f}")
big3 = ev[(ev.peak_ex >= 0.3)]; print(f"очень крупные (не менее 30 процентов): {len(big3)} ({len(big3)/L.day.nunique()*30:.1f} на 30 дней), обнаружены {(big3.detect >= 0.5).mean():.2f}")
pk = ev[ev.hr.isin([7, 8, 9, 16, 17, 18, 19])]; print(f"в часы пик: {len(pk)} событий, обнаружены {(pk.detect >= 0.5).mean():.2f}")
ev.to_csv("surge_events.csv")
