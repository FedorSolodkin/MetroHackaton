"""Как модель (LightGBM календарь+лаги, прогноз за 30 мин, месяц в проверку) ведёт себя именно на пиках.
Смотрим: высоту и время пиков, смещение прогноза (занижает ли), калибровку по верхним децилям, пропущенные всплески.
Вход: residuals.pkl (ts, station, pax, p), surge_events.csv. Запуск из C:\metro."""
import numpy as np, pandas as pd
r = pd.read_pickle("residuals.pkl"); r["day"] = (r.ts - pd.Timedelta(hours=3)).dt.normalize(); r["hr"] = r.ts.dt.hour; r["dow"] = r.ts.dt.dayofweek
wape = lambda d: np.abs(d.pax - d.p).sum() / d.pax.sum()
L = r.groupby(["ts", "day", "dow", "hr"]).agg(pax=("pax", "sum"), p=("p", "sum")).reset_index()
print("=== 1. ЛИНИЯ ЦЕЛИКОМ, суточные пики (будни, без праздников) ===")
wd = L[(L.dow < 5)]; rows = []
for (d, win), g in {(d, w): x for d, g0 in wd.groupby("day") for w, x in (("утро", g0[g0.hr.between(6, 11)]), ("вечер", g0[g0.hr.between(15, 20)])) if len(x) > 8}.items():
    ia, ip = g.pax.idxmax(), g.p.idxmax()
    rows.append((d, win, g.pax.max(), g.p.max(), (g.loc[ip, "ts"] - g.loc[ia, "ts"]).total_seconds() / 60, g.loc[ia, "p"] / g.pax.max() - 1))
P = pd.DataFrame(rows, columns=["day", "win", "act_max", "pred_max", "dt_min", "pred_at_act_peak_err"])
P["height_err"] = P.pred_max / P.act_max - 1
for win in ("утро", "вечер"):
    q = P[P.win == win]
    print(f"{win}: дней {len(q)} | высота пика прогноза к факту: среднее {q.height_err.mean():+.1%}, |ошибка| медиана {q.height_err.abs().median():.1%}, занижен в {(q.height_err < 0).mean():.0%} дней "
          f"| время пика: попали в ±15 мин {(q.dt_min.abs() <= 15).mean():.0%}, ±30 мин {(q.dt_min.abs() <= 30).mean():.0%} | в слоте факт-пика прогноз {q.pred_at_act_peak_err.mean():+.1%}")
print("\n=== 2. ВСЕ СТАНЦИИ: верхние слоты (верхний дециль потока станции в своём типе дня и часе) ===")
r["wk"] = r.dow >= 5
r["q"] = r.groupby(["station", "wk", "hr"]).pax.transform(lambda s: s.rank(pct=True))
for lab, mk in (("все слоты", r.q > 0), ("верхние 10% по станции/часу", r.q > 0.9), ("верхние 2%", r.q > 0.98), ("нижние 50%", r.q <= 0.5)):
    d = r[mk & (r.p > 50)]; print(f"  {lab:30s} n={len(d):6d} WAPE={wape(d):.1%}  смещение (прогноз−факт)/факт={((d.p - d.pax).sum() / d.pax.sum()):+.1%}  занижено в {(d.p < d.pax).mean():.0%} слотов")
print("\n  калибровка: наклон факт~прогноз по децилям прогноза (1.00 = без сжатия)")
r["pq"] = pd.qcut(r.p, 10, labels=False, duplicates="drop"); c = r.groupby("pq").agg(p=("p", "mean"), a=("pax", "mean")); c["a/p"] = c.a / c.p
print("  " + " ".join(f"D{int(i)+1}:{v:.2f}" for i, v in c["a/p"].items()))
print("\n=== 3. ПРОПУЩЕННЫЕ ВСПЛЕСКИ (линия, +10% к норме, >=30 мин, без праздников) ===")
ev = pd.read_csv("surge_events.csv", parse_dates=["day", "start"]); ev["found"] = ev.detect >= 0.5
print(f"всего {len(ev)}, найдено {ev.found.mean():.0%}, пропущено {(~ev.found).sum()}")
print("доля найденных по размеру (макс. превышение):", {f"{lo:.0%}-{hi:.0%}": round(ev[(ev.peak_ex >= lo) & (ev.peak_ex < hi)].found.mean(), 2) for lo, hi in ((0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 9))})
ev["dowl"] = ev.day.dt.dayofweek; print("пропущенные по дню недели:", ev[~ev.found].groupby("dowl").size().to_dict(), "| по часу начала:", ev[~ev.found].groupby("hr").size().sort_values(ascending=False).head(5).to_dict())
print("длительность: найденные %.0f мин, пропущенные %.0f мин (медиана)" % (ev[ev.found].slots.median() * 15, ev[~ev.found].slots.median() * 15))
print("\n10 крупнейших пропущенных всплесков:"); print(ev[~ev.found].sort_values("extra_pax", ascending=False).head(10)[["start", "slots", "extra_pax", "peak_ex", "detect"]].round(2).to_string(index=False))
print("\n10 крупнейших найденных:"); print(ev[ev.found].sort_values("extra_pax", ascending=False).head(5)[["start", "slots", "extra_pax", "peak_ex", "detect"]].round(2).to_string(index=False))
