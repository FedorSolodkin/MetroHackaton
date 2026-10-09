"""Нагрузка от мероприятий по станциям Линии 1 с шагом 15 минут.

Идея: много мелких событий рядом с одной станцией суммируются. Для каждого события
оцениваем число людей, которые поедут на метро, раскладываем по времени (приход до
начала, уход после конца) и по ближайшим станциям, затем суммируем по (станция, слот).

Запуск:
    python station_load.py events_raw.csv 2026-02-01 2026-09-30  ->  event_features.csv
Допущения (все параметры внизу файла можно менять и калибровать по остаткам модели).
"""
import re
import sys
from math import exp

import numpy as np
import pandas as pd

from stations import STATIONS, nearest_stations

# --- допущения -------------------------------------------------------------
FILL = {"sport": 0.9, "default": 0.7}           # заполняемость
METRO_SHARE = 0.40                              # доля приехавших/уехавших на метро
RADIUS_KM = 1.5                                 # радиус привязки к станции
DECAY_KM = 0.6                                  # затухание веса станции с расстоянием
ARRIVE_MIN = 60                                 # приход людей за N минут до начала
LEAVE_SHARES = [0.5, 0.3, 0.2]                  # уход: доли в первые три слота после конца
CONTINUOUS_H = 8                                # дольше этого событие считается "фоновым"
CONT_RATE = 0.30                                # доля вместимости, проходящая за время фонового события

CAP_DEFAULT = {"sport": 15000, "concert": 1500, "theater": 700, "exhibition": 300,
               "festival": 3000, "cinema": 150, "kids": 200, "other": 80}
# вместимость по названию площадки (подстрока, нижний регистр). Значения приблизительные, проверить.
CAP_BY_VENUE = {"газпром арена": 56000, "октябрьский": 3800, "ска арена": 21500,
                "ледовый дворец": 12300}

# Площадки далеко от Линии 1: люди добираются через другие линии и пересадки.
# {подстрока названия: [(станция Линии 1, доля), ...]}. Предположение, проверить по остаткам.
VENUE_FEEDERS = {"газпром арена": [("Пушкинская", 0.5), ("Владимирская", 0.5)]}


def _cat(c: str) -> str:
    c = (c or "other").lower()
    for k in CAP_DEFAULT:
        if k in c:
            return k
    if "спорт" in c:
        return "sport"
    if "концерт" in c:
        return "concert"
    if "театр" in c:
        return "theater"
    if "выстав" in c:
        return "exhibition"
    return "other"


def _weights(row):
    name = str(row["venue"]).lower()
    for key, feed in VENUE_FEEDERS.items():
        if key in name:
            return feed
    near = nearest_stations(row["lat"], row["lon"], RADIUS_KM)
    if not near:
        return []
    w = np.array([exp(-d / DECAY_KM) for _, d in near])
    w = w / w.sum()
    return [(s, float(x)) for (s, _), x in zip(near, w)]


def _people(row):
    cap = row.get("capacity")
    if pd.isna(cap):
        name = str(row["venue"]).lower()
        cap = next((v for k, v in CAP_BY_VENUE.items() if k in name), None)
    cat = _cat(row["category"])
    if cap is None or pd.isna(cap):
        cap = CAP_DEFAULT[cat]
    return cap * FILL.get(cat, FILL["default"]) * METRO_SHARE


def build(events: pd.DataFrame, since: str, until: str) -> pd.DataFrame:
    grid = pd.date_range(since, pd.Timestamp(until) + pd.Timedelta(days=1), freq="15min", inclusive="left")
    idx = {t: i for i, t in enumerate(grid)}
    st = [s for s, _, _ in STATIONS]
    sidx = {s: i for i, s in enumerate(st)}
    n = len(grid)
    ev_in, ev_out, ev_cont, ev_n = (np.zeros((len(st), n)) for _ in range(4))

    def slot(ts):
        return idx.get(pd.Timestamp(ts).floor("15min"))

    for _, r in events.iterrows():
        ws = _weights(r)
        if not ws:
            continue
        p = _people(r)
        s, e = pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
        dur_h = (e - s).total_seconds() / 3600
        for stn, w in ws:
            i = sidx[stn]
            if dur_h > CONTINUOUS_H:
                a, b = slot(s) if slot(s) is not None else 0, slot(e)
                a = max(a, 0)
                b = n - 1 if b is None else b
                if b >= a:
                    ev_cont[i, a:b + 1] += p * w * CONT_RATE / max(b - a + 1, 1)
                continue
            k = ARRIVE_MIN // 15
            for j in range(1, k + 1):                      # приход
                t = slot(s - pd.Timedelta(minutes=15 * j))
                if t is not None:
                    ev_in[i, t] += p * w / k
            for j, sh in enumerate(LEAVE_SHARES):          # уход
                t = slot(e + pd.Timedelta(minutes=15 * j))
                if t is not None:
                    ev_out[i, t] += p * w * sh
            a, b = slot(s), slot(e)
            if a is not None and b is not None:
                ev_n[i, a:b + 1] += 1

    rows = []
    for stn, i in sidx.items():
        rows.append(pd.DataFrame({"ts": grid, "station": stn, "ev_in": ev_in[i], "ev_out": ev_out[i],
                                  "ev_cont": ev_cont[i], "ev_n": ev_n[i]}))
    out = pd.concat(rows, ignore_index=True)
    out["ev_total"] = out.ev_in + out.ev_out + out.ev_cont
    return out


if __name__ == "__main__":
    ev = pd.read_csv(sys.argv[1], parse_dates=["start", "end"])
    feat = build(ev, sys.argv[2], sys.argv[3])
    feat.to_csv("event_features.csv", index=False)
    print("записано", len(feat), "строк; ненулевых:", int((feat.ev_total > 0).sum()))
