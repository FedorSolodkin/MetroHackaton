"""Выгрузка мероприятий Санкт-Петербурга из KudaGo и Timepad + ручные расписания.

Запуск (там, где открываются kudago.com и api.timepad.ru):
    set TIMEPAD_TOKEN=...           (токен Timepad, необязательно)
    python fetch_events.py 2026-02-01 2026-09-30

Результат: events_raw.csv со столбцами
    source, title, category, venue, lat, lon, start, end, capacity
Ручные события берутся из manual_events.csv (те же столбцы, кроме source).
"""
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

KUDAGO = "https://kudago.com/public-api/v1.4"
TIMEPAD = "https://api.timepad.ru/v1/events"
FOREVER = 4_000_000_000  # у KudaGo бывают "бесконечные" даты


def _ts(d: str) -> int:
    return int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp())


def _get(url, params=None, headers=None, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=40)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):
                time.sleep(2 * (i + 1))
                continue
            print("HTTP", r.status_code, url)
            return None
        except requests.RequestException as e:
            print("ошибка запроса:", e)
            time.sleep(2 * (i + 1))
    return None


def fetch_kudago(since: str, until: str) -> pd.DataFrame:
    params = {
        "location": "spb",
        "actual_since": _ts(since),
        "actual_until": _ts(until) + 86400,
        "page_size": 100,
        "fields": "id,title,dates,place,categories",
        "expand": "place",
        "text_format": "text",
    }
    rows, url, first = [], f"{KUDAGO}/events/", True
    while url:
        j = _get(url, params if first else None)
        first = False
        if not j:
            break
        for e in j.get("results", []):
            p = e.get("place") or {}
            c = p.get("coords") or {}
            if not c.get("lat"):
                continue  # без координат на станцию не привяжем
            cat = (e.get("categories") or ["other"])[0]
            for d in e.get("dates", []):
                s, en = d.get("start"), d.get("end")
                if s is None or s < 0 or s >= FOREVER:
                    continue
                if en is None or en >= FOREVER:
                    en = s + 3 * 3600
                rows.append(("kudago", e["title"], cat, p.get("title"), c["lat"], c["lon"],
                             datetime.fromtimestamp(s, timezone.utc), datetime.fromtimestamp(en, timezone.utc), None))
        url = j.get("next")
        time.sleep(0.3)
    print("KudaGo:", len(rows))
    return _frame(rows)


def fetch_timepad(since: str, until: str, token: str) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {token}"}
    rows, skip = [], 0
    while True:
        j = _get(TIMEPAD, {
            "cities": "Санкт-Петербург", "starts_at_min": since + "T00:00:00",
            "starts_at_max": until + "T23:59:59", "limit": 100, "skip": skip,
            "fields": "location,ends_at,starts_at,categories", "sort": "+starts_at",
        }, headers)
        vals = (j or {}).get("values", [])
        if not vals:
            break
        for e in vals:
            loc = e.get("location") or {}
            co = loc.get("coordinates")
            if isinstance(co, str):
                co = [float(x) for x in co.replace(" ", "").split(",")]
            if not co:
                continue
            s = pd.Timestamp(e["starts_at"]).tz_convert("UTC")
            en = pd.Timestamp(e["ends_at"]).tz_convert("UTC") if e.get("ends_at") else s + pd.Timedelta(hours=3)
            cat = (e.get("categories") or [{"name": "other"}])[0].get("name", "other")
            rows.append(("timepad", e.get("name", ""), cat, loc.get("address"), co[0], co[1], s, en, None))
        skip += len(vals)
        time.sleep(0.3)
    print("Timepad:", len(rows))
    return _frame(rows)


def load_manual(path="manual_events.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return _frame([])
    d = pd.read_csv(path, parse_dates=["start", "end"])
    for c in ("start", "end"):
        d[c] = d[c].dt.tz_localize("Europe/Moscow").dt.tz_convert("UTC")
    d["source"] = "manual"
    return d[COLS]


COLS = ["source", "title", "category", "venue", "lat", "lon", "start", "end", "capacity"]


def _frame(rows):
    return pd.DataFrame(rows, columns=COLS)


if __name__ == "__main__":
    since, until = sys.argv[1], sys.argv[2]
    parts = [fetch_kudago(since, until), load_manual()]
    tok = os.environ.get("TIMEPAD_TOKEN")
    if tok:
        parts.append(fetch_timepad(since, until, tok))
    else:
        print("TIMEPAD_TOKEN не задан, Timepad пропущен")
    ev = pd.concat(parts, ignore_index=True)
    ev["start"] = pd.to_datetime(ev.start, utc=True).dt.tz_convert("Europe/Moscow").dt.tz_localize(None)
    ev["end"] = pd.to_datetime(ev.end, utc=True).dt.tz_convert("Europe/Moscow").dt.tz_localize(None)
    ev.to_csv("events_raw.csv", index=False)
    print("всего событий (с датами):", len(ev))
