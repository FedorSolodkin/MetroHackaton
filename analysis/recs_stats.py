"""Офлайн-статистика рекомендаций на реальных днях: «из N дней — сколько рекомендаций, сколько подтвердилось».
Для каждого дня каждые 30 минут (05:45–23:00) строится состояние, как в приложении (ансамбль, модели не видели месяц дня); сравниваем рекомендацию «добавить составы»
с тем, что случилось на самом деле: фактический вход следующего часа даёт загрузку перегонов; «подтверждено», если там же загрузка >= 88% комфортной вместимости и вход >= +8% к норме
(то же условие, что и у рекомендации, но по факту, а не по прогнозу).
Запуск из корня репозитория:  python analysis/recs_stats.py [шаг_в_слотах=2]   (METRO_CHRONOS=0 — без Chronos-2, быстрее)"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
import decide as D  # noqa: E402
import engine as E  # noqa: E402
from ensemble import Ensemble  # noqa: E402

STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 2
W = E.World(); W.ens = Ensemble(W)
print("ансамбль:", W.ens.names, "|", W.ens.chronos_note, flush=True)
valid = (~np.isnan(W.X).any(axis=1)) & np.isin(W.month, list(W.models))
DAYS = [d for d in sorted(set(W.svc[valid])) if (W.svc == d).sum() >= 90]
SEC = D.SECTIONS
LIM = int(os.environ.get("DAYS_LIMIT", 0))
if LIM: DAYS = DAYS[::max(1, len(DAYS) // LIM)][:LIM]


def idx_at(day, hm):
    return E.GRID.get_loc(pd.Timestamp(day) + pd.Timedelta(hours=int(hm[:2]), minutes=int(hm[3:]))) - 1


def truth(c, d, sec, P):
    """фактическая загрузка и отклонение перегонов участка за ближайший час (слоты c+1..c+4): (макс. загрузка, отклонение сегмента с макс. загрузкой среди отклонившихся >= +8%)"""
    we = bool(W.weekend_day[c]); hour = E.GRID[c + 1].hour; phase = "we" if we else ("am" if hour < 13 else "pm")
    A = W.X[c + 1:c + 5].sum(0); N = np.array([W.norm_adj(c + k) for k in range(1, 5)]).sum(0)
    lp, ln = W.seg_loads(A, phase), W.seg_loads(np.where(np.isnan(N), A, N), phase); i = 0 if d == "to_veteranov" else 1
    k0, k1, _, _ = SEC[sec]; seg = np.arange(k0, k1 + 1); u = lp[i][seg] / (P * E.CAP_TRAIN); dev = lp[i][seg] / np.clip(ln[i][seg], 1, None) - 1
    cand = np.where(dev >= E.DEV_UP)[0]; k = int(cand[np.argmax(u[cand])]) if len(cand) else int(np.argmax(u))
    return float(u[k]), float(dev[k]), float(u.max())


rows, t0 = [], time.time()
for n, day in enumerate(DAYS):
    month = int(W.month[idx_at(day, "05:45")]); we = bool(W.weekend_day[idx_at(day, "05:45")])
    for c in range(idx_at(day, "05:45"), idx_at(day, "23:00") + 1, STEP):
        if c + 5 >= E.T or np.isnan(W.NORM[c + 1]).any():
            continue
        st = D.build_state(W, c, W.X, str(pd.Timestamp(day).date()), {})
        P = float(np.mean([E.plan_pairs(E.GRID[c + 1 + k].hour, month, we) for k in range(4)]))
        got = {(r["direction"], r["section"]): r for r in st["recommendations"] if r["action"] in ("up", "limit")}
        for d in ("to_veteranov", "to_devyatkino"):
            for sec in SEC:
                u, dev, umax = truth(c, d, sec, P); real = (u >= E.U_HI) and (dev >= E.DEV_UP)
                r = got.get((d, sec))
                rows.append(dict(day=str(pd.Timestamp(day).date()), time=E.hhmm(E.GRID[c] + pd.Timedelta(minutes=15)), direction=d, section=sec, rec=r is not None, action=r["action"] if r else "",
                                 trains=r["trains_delta"] if r else 0, delta_s=r["delta_s"] if r else 0, real=real, u_fact=u, dev_fact=dev, over100=u >= 1.0))
    if n % 10 == 9:
        print(f"{n + 1}/{len(DAYS)} дней, {time.time() - t0:.0f} с", flush=True)
df = pd.DataFrame(rows); os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True); df.to_csv(os.path.join(ROOT, "reports", "recs_stats.csv"), index=False, encoding="utf-8")


def episodes(mask_df, col):
    """эпизоды = подряд идущие шаги (по дню/направлению/участку) с признаком col"""
    n = 0
    for _, g in mask_df.groupby(["day", "direction", "section"]):
        v = g[col].values.astype(int); n += int(((v[1:] == 1) & (v[:-1] == 0)).sum() + (v[0] == 1))
    return n


rec_ep = episodes(df, "rec"); real_ep = episodes(df, "real")
tp = int((df.rec & df.real).sum()); fp = int((df.rec & ~df.real).sum()); fn = int((~df.rec & df.real).sum()); days_rec = df[df.rec].day.nunique(); days_real = df[df.real].day.nunique()
out = [f"Дней: {len(DAYS)}; шаг {STEP * 15} мин; всего шагов×направлений×участков: {len(df)}",
       f"Рекомендации «добавить/предел»: {int(df.rec.sum())} шагов, эпизодов {rec_ep}, дней с рекомендацией {days_rec} из {len(DAYS)}",
       f"Фактически (по следующему часу) условие выполнилось: {int(df.real.sum())} шагов, эпизодов {real_ep}, дней {days_real}",
       f"Подтверждено рекомендаций (факт тоже >= 88% и >= +8%): {tp} из {int(df.rec.sum())} шагов = {tp / max(df.rec.sum(), 1) * 100:.0f}% (точность)",
       f"Пойманных случаев факта: {tp} из {tp + fn} = {tp / max(tp + fn, 1) * 100:.0f}% (полнота)",
       f"Из рекомендаций перегруз выше 100% по факту: {int((df.rec & df.over100).sum())} шагов",
       f"Ложных: {fp} шагов; пропусков: {fn} шагов"]
print("\n".join(out)); open(os.path.join(ROOT, "reports", "RECS_STATS.md"), "w", encoding="utf-8").write("# Статистика рекомендаций на реальных днях\n\n" + "\n\n".join("- " + x for x in out) + "\n")
