"""Проверка демо: что прогноз говорил в каждый момент и что было в следующий час (вход по линии, загрузка худшего перегона).
Запуск из корня репозитория:  METRO_CHRONOS=0 python analysis/demo_check.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
import decide as D  # noqa: E402
import engine as E  # noqa: E402
from ensemble import Ensemble  # noqa: E402
from minutes import MinuteFeed  # noqa: E402

W = E.World(); W.ens = Ensemble(W)


def at(day, hm):
    return pd.Timestamp(day) + pd.Timedelta(hours=int(hm[:2]), minutes=int(hm[3:]))


def run(day, inj, times, title):
    F = MinuteFeed(W, day, inj); print("==", title)
    for hm in times:
        t = at(day, hm); m = int((t - E.GRID[0]) / pd.Timedelta(minutes=1)); c, off = (m - 15) // 15, (m - 15) % 15
        st = D.build_state(W, c, F.x_roll(W.X, c, off), day, {}, {}, off=off, feed=F)
        fc = sum(st["series"]["forecast"][8:12]); a = F.minute_index(t); e = F.counts(a, a + 60); fact = e.sum()
        ph = "we" if W.weekend_day[c] else ("am" if t.hour < 13 else "pm"); s_, n_ = W.seg_loads(e, ph)
        cap = np.mean([E.plan_pairs((t + pd.Timedelta(minutes=15 * k)).hour, int(W.month[c]), bool(W.weekend_day[c])) for k in range(4)]) * E.CAP_TRAIN
        u_fc = max(x["u"] for d in st["segments"].values() for x in d); u_fact = max(s_.max(), n_.max()) / cap
        print(f"{hm}  вход за час: ошибка {(fc - fact) / fact * 100:+6.1f}%  | загрузка худшего перегона: прогноз {u_fc*100:4.0f}%  факт {u_fact*100:4.0f}%  | рек: {[r['action'] for r in st['recommendations']]}")


N4 = ["Площадь Мужества", "Политехническая", "Академическая", "Гражданский проспект"]
run("2026-09-12", [{"stations": N4, "start": "12:30", "end": "15:00", "pct": 240, "ramp": 45, "decay": 45}],
    ["12:15", "12:45", "13:00", "13:05", "13:10", "13:15", "13:30", "13:45", "14:00", "14:30", "15:00", "15:15", "15:30"], "демо-день 12.09 (модельный приток)")
#run("2026-05-12", [], ["08:00", "12:00", "17:00", "18:00"], "обычный вторник 12.05")
