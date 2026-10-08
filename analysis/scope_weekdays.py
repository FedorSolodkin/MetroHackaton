"""Оценка в рамках: обычные рабочие дни (пн-пт, без праздников и предпраздничных), без часа открытия (05:30-06:30), без ночи.
Сравнение: медиана (база) / LightGBM, обученный на всех днях / LightGBM, обученный только в рамках. Пики и всплески линии.
Вход: model_table.pkl. Запуск из C:\metro."""
import numpy as np, pandas as pd, lightgbm as lgb
a = pd.read_pickle("model_table.pkl")
scope = (a.dow < 5) & (a.off == 0) & (a.pre == 0) & (a.slot >= 14) & (a.slot <= 85)
print(f"в рамках: {scope.sum():,} из {len(a):,} строк ({scope.mean():.0%}); дней: {a[scope].day.nunique()}")
base = ["st", "slot", "dow", "off", "pre", "off_next", "off_prev", "lag2", "lag3", "lag4", "lag6", "lag8", "lag96", "lag672", "mean_2_5", "trend"]
P = dict(objective="regression_l1", learning_rate=0.08, num_leaves=63, min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
a["typ"] = np.where(a.dow == 4, "fri", "wd")
wape = lambda y, q: np.abs(y - q).sum() / y.sum(); out = []
for m in (2, 5, 7, 9):
    te = a[(a.m == m) & scope].copy()
    med = a[(a.m != m) & scope].groupby(["station", "typ", "slot"]).pax.median().rename("med").reset_index()
    te = te.merge(med, on=["station", "typ", "slot"], how="left")
    for n, trm in (("LGBM все дни", a.m != m), ("LGBM только рамки", (a.m != m) & scope)):
        mdl = lgb.train(P, lgb.Dataset(a[trm][base], a[trm].pax, categorical_feature=["st"]), 300); te[n] = mdl.predict(te[base])
    out.append(te[["ts", "day", "m", "station", "pax", "med", "LGBM все дни", "LGBM только рамки", "slot"]])
te = pd.concat(out).dropna(subset=["med"]); te.to_pickle("scope_preds.pkl")
names = ["med", "LGBM все дни", "LGBM только рамки"]
tab = pd.DataFrame({n: {m: wape(te[te.m == m].pax.values, te[te.m == m][n].values) * 100 for m in (2, 5, 7, 9)} for n in names}).T
tab["среднее"] = tab.mean(axis=1); print("\nWAPE, % (только рамки, месяц в проверку):"); print(tab.round(2).rename(index={"med": "медиана (база)"}).rename(columns={2: "фев", 5: "май", 7: "июль", 9: "сен"}).to_string())
hr = (te.slot * 15 + 180) // 60 % 24; pk = hr.isin([7, 8, 9, 16, 17, 18])
print("\nпиковые часы (7–9, 16–18): " + " | ".join(f"{n}={wape(te[pk].pax.values, te[pk][n].values) * 100:.2f}" for n in names))
print("внепиковые:                " + " | ".join(f"{n}={wape(te[~pk].pax.values, te[~pk][n].values) * 100:.2f}" for n in names))
best = "LGBM только рамки" if tab.loc["LGBM только рамки", "среднее"] < tab.loc["LGBM все дни", "среднее"] else "LGBM все дни"; print("лучшая модель в рамках:", best)
# линия целиком: суточные пики
L = te.groupby(["ts", "day", "m"]).agg(pax=("pax", "sum"), p=(best, "sum"), med=("med", "sum")).reset_index(); L["hr"] = L.ts.dt.hour
rows = []
for d, g0 in L.groupby("day"):
    for w, g in (("утро", g0[g0.hr.between(6, 11)]), ("вечер", g0[g0.hr.between(15, 20)])):
        if len(g) > 8:
            ia, ip = g.pax.idxmax(), g.p.idxmax(); rows.append((w, g.p.max() / g.pax.max() - 1, (g.loc[ip, "ts"] - g.loc[ia, "ts"]).total_seconds() / 60))
Pk = pd.DataFrame(rows, columns=["w", "h", "dt"]); print("\nсуточные пики линии:")
for w, q in Pk.groupby("w"): print(f"  {w}: дней {len(q)} | высота прогноза к факту {q.h.mean():+.1%} (|ошибка| медиана {q.h.abs().median():.1%}) | время ±15 мин {(q.dt.abs() <= 15).mean():.0%}")
# всплески линии (+10% к норме, >=30 мин) относительно медианы
THR = 0.10; L = L.sort_values("ts"); L["ex"] = L.pax / L.med - 1; L["s"] = L.ex > THR
L["run"] = (L.s != L.s.shift()) | (L.day != L.day.shift()); L["rid"] = L.run.cumsum(); rows = []
for rid, g in L[L.s].groupby("rid"):
    if len(g) >= 2: rows.append(dict(start=g.ts.min(), day=g.day.iloc[0], m=g.m.iloc[0], slots=len(g), extra=(g.pax - g.med).sum(), peak=g.ex.max(), det=((g.p - g.med) >= 0.5 * (g.pax - g.med)).mean()))
E = pd.DataFrame(rows); E["found"] = E.det >= 0.5; nd = L.day.nunique()
print(f"\nвсплески (+10% к норме, >=30 мин): {len(E)} за {nd} раб. дней ({len(E)/nd:.2f} в день, ≈{len(E)/nd*21.5:.0f} в месяц из 21,5 раб. дней)")
print(f"  найдено за 30 мин: {E.found.mean():.0%}; по размеру: " + ", ".join(f"{lo:.0%}–{hi:.0%}: {E[(E.peak >= lo) & (E.peak < hi)].found.mean():.0%} (n={((E.peak >= lo) & (E.peak < hi)).sum()})" for lo, hi in ((0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 9))))
big = E[E.peak >= 0.2]; print(f"  крупные (>=20%): {len(big)} (≈{len(big)/nd*21.5:.0f} в месяц), найдено {big.found.mean():.0%}")
print("  пропущено:", int((~E.found).sum()), "| по часу начала:", E[~E.found].start.dt.hour.value_counts().sort_index().to_dict())
print("\n8 крупнейших пропущенных:"); print(E[~E.found].sort_values("extra", ascending=False).head(8)[["start", "slots", "extra", "peak", "det"]].round(2).to_string(index=False))
E.to_csv("surge_events_scope.csv", index=False)
