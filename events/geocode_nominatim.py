"""Геокодирование адресов Timepad через Nominatim (OSM). Лимит сервиса: 1 запрос/сек.
python geocode_nominatim.py  -> geocode_cache.csv (address_clean, lat, lon, display); можно прерывать и продолжать."""
import csv, os, re, time
import pandas as pd, requests

H = {"User-Agent": "MetroHackaton/1.0 (student hackathon project)"}
URL = "https://nominatim.openstreetmap.org/search"; CACHE = "geocode_cache.csv"

def clean(a: str) -> str:
    a = re.sub(r"\s+", " ", str(a).replace("\r", " ").replace("\n", " ")).strip()
    a = re.sub(r"\(.*?\)", " ", a)                          # скобки: этаж, пояснения
    a = re.sub(r"(ориентир|вход|этаж|эт\.|офис|оф\.|помещение|пом\.|зал)\b.*$", "", a, flags=re.I)
    a = re.sub(r"\s+", " ", a).strip(" ,.;")
    if not re.search(r"санкт|спб|петербург", a, re.I):
        a += ", Санкт-Петербург"
    return a

def geocode(q):
    for _ in range(3):
        try:
            r = requests.get(URL, params=dict(q=q, format="jsonv2", limit=1, countrycodes="ru",
                             viewbox="29.4,60.3,30.9,59.6", bounded=1), headers=H, timeout=30)
            if r.status_code == 200:
                j = r.json()
                return (float(j[0]["lat"]), float(j[0]["lon"]), j[0]["display_name"][:80]) if j else (None, None, "")
            time.sleep(5)
        except requests.RequestException:
            time.sleep(5)
    return (None, None, "error")

if __name__ == "__main__":
    t = pd.read_csv("timepad_raw.csv")
    t = t[t.lat.isna() & t.venue.notna()]
    t["q"] = t.venue.map(clean)
    order = t.q.value_counts().index.tolist()                # сначала самые частые
    done = set(pd.read_csv(CACHE).q) if os.path.exists(CACHE) else set()
    new = not os.path.exists(CACHE)
    with open(CACHE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(["q", "lat", "lon", "display"])
        for i, q in enumerate(order):
            if q in done: continue
            lat, lon, d = geocode(q); w.writerow([q, lat, lon, d]); f.flush()
            if i % 100 == 0: print(i, "/", len(order), flush=True)
            time.sleep(1.1)
    print("готово")
