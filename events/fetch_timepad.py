"""Выгрузка событий Timepad по Санкт-Петербургу (токен в TIMEPAD_TOKEN).
python fetch_timepad.py 2026-02-01 2026-02-28 [...пары дат]  -> timepad_raw.csv
Сохраняет события и без координат (адрес остаётся для последующего геокодирования)."""
import os, sys, time
import pandas as pd, requests

H = {"Authorization": "Bearer " + os.environ["TIMEPAD_TOKEN"]}; U = "https://api.timepad.ru/v1/events"

def month(a, b):
    rows, skip = [], 0
    while True:
        r = requests.get(U, headers=H, timeout=60, params={"limit": 100, "skip": skip, "cities": "Санкт-Петербург",
            "starts_at_min": a + "T00:00:00", "starts_at_max": b + "T23:59:59", "sort": "+starts_at",
            "fields": "location,ends_at,starts_at,categories"})
        if r.status_code != 200:
            print("HTTP", r.status_code, r.text[:150]); break
        v = r.json().get("values", [])
        if not v: break
        for e in v:
            loc = e.get("location") or {}; co = loc.get("coordinates")
            lat, lon = (float(co[0]), float(co[1])) if co and len(co) == 2 else (None, None)
            s = pd.Timestamp(e["starts_at"]); en = pd.Timestamp(e["ends_at"]) if e.get("ends_at") else s + pd.Timedelta(hours=3)
            cat = (e.get("categories") or [{"name": "other"}])[0].get("name", "other")
            rows.append(("timepad", e.get("name", ""), cat, loc.get("address"), lat, lon,
                         s.tz_convert("Europe/Moscow").tz_localize(None), en.tz_convert("Europe/Moscow").tz_localize(None), None))
        skip += len(v); time.sleep(0.25)
    return rows

if __name__ == "__main__":
    a = sys.argv[1:]; rows = []
    for i in range(0, len(a), 2):
        r = month(a[i], a[i + 1]); print(a[i], a[i + 1], len(r), flush=True); rows += r
    pd.DataFrame(rows, columns=["source", "title", "category", "venue", "lat", "lon", "start", "end", "capacity"]).to_csv("timepad_raw.csv", index=False)
