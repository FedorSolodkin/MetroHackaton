"""Поезда в эмуляторе (симуляция): положение по интервалу движения и загрузка на каждой станции — как данные о массе вагонов (РПДП).
Отправления с конечных — с шагом 3600 / пары в час (график, заданный вручную интервал, принятые составы). Посадка на станции = вход
за время с прошлого поезда этого направления; куда едут — по модели корреспонденций (engine.World.W), высадка — на станции назначения."""
import numpy as np
import pandas as pd

import engine as E

DIRS = ("to_veteranov", "to_devyatkino")
WINDOW_MIN = 30          # загрузка перегона «по данным поездов» = средняя загрузка составов, прошедших его за последние 30 минут


def _order(d):
    return list(range(E.NS - 1, -1, -1)) if d == "to_veteranov" else list(range(E.NS))


def departures(pairs, d, day, t_to):
    """времена отправления с конечной с 05:30 до t_to"""
    t = pd.Timestamp(day) + pd.Timedelta(hours=5, minutes=30); out = []
    while t <= t_to and len(out) < 2000:
        out.append(t); p = pairs(d, t)
        t = t + pd.Timedelta(seconds=3600 / p) if p > 0.5 else t + pd.Timedelta(minutes=20)
    return out


def simulate(w, feed, t_now, pairs, phase, day):
    Wp = w.W[phase]; P = Wp / Wp.sum(1, keepdims=True); hop = pd.Timedelta(minutes=E.HOP_MIN)
    trains, meas, reports = [], {}, {}
    for d in DIRS:
        order = _order(d); deps = departures(pairs, d, day, t_now)
        seg = [[] for _ in range(E.NS - 1)]
        first = next((i for i, t in enumerate(deps) if t >= t_now - pd.Timedelta(minutes=E.HOP_MIN * (E.NS - 1) + WINDOW_MIN + 5)), len(deps))
        for i in range(max(first, 1), len(deps)):
            dep, prev = deps[i], deps[i - 1]; onboard = np.zeros(E.NS); last = None
            for j, st in enumerate(order):
                t_arr = dep + j * hop
                if t_arr > t_now: break
                onboard[st] = 0.0                                                     # высадка: кто ехал до этой станции
                ent = feed.counts(feed.minute_index(prev + j * hop), feed.minute_index(t_arr))[st] * w.scale
                down = order[j + 1:]
                if down: onboard[down] += ent * P[st, down]                          # посадка: в сторону движения, по назначениям
                load = float(onboard.sum()); last = (st, j, load, t_arr)
                if down:
                    k = st - 1 if d == "to_veteranov" else st; seg[k].append((t_arr, load))
                    reports[(d, st)] = {"train": i + 1, "load": round(load), "time": E.hhmm(t_arr)}
            if last is not None and dep + (E.NS - 1) * hop >= t_now:                # поезд сейчас на линии
                st, j, load, t_arr = last; frac = min(1.0, (t_now - t_arr) / hop)
                pos = st - frac if d == "to_veteranov" else st + frac
                trains.append({"id": f"{'Ю' if d == 'to_veteranov' else 'С'}{i + 1}", "direction": d, "pos": round(float(pos), 3), "load": round(load),
                               "u": round(load / E.CAP_TRAIN, 3), "near": E.ST[st]})
        t0 = t_now - pd.Timedelta(minutes=WINDOW_MIN)
        meas[d] = np.array([np.mean([l for t, l in seg[k] if t > t0]) if any(t > t0 for t, _ in seg[k]) else np.nan for k in range(E.NS - 1)])
    return {"trains": trains, "seg_train_load": meas, "reports": reports}
