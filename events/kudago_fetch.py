"""Выгрузка мероприятий Санкт-Петербурга из KudaGo за февраль, май, июль и сентябрь 2026.
Запуск: двойной клик по kudago_fetch.exe (VPN лучше отключить). Результат: kudago_events.csv рядом с exe.
"""
import csv, os, sys, time
from datetime import datetime, timezone
import requests

BASE = "https://kudago.com/public-api/v1.4/events/"
PERIODS = [("2026-02-01", "2026-02-28"), ("2026-05-01", "2026-05-31"), ("2026-07-01", "2026-07-31"), ("2026-09-01", "2026-09-30")]
FOREVER = 4_000_000_000
OUT = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "kudago_events.csv")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36", "Accept": "application/json"}


def ts(d, end=False):
    return int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp()) + (86400 if end else 0)


def get(session, url, params=None):
    for i in range(4):
        try:
            r = session.get(url, params=params, headers=UA, timeout=45)
            if r.status_code == 200:
                return r.json()
            print(f"  HTTP {r.status_code}, повтор {i + 1}/4")
        except requests.RequestException as e:
            print(f"  ошибка соединения ({type(e).__name__}), повтор {i + 1}/4")
        time.sleep(3 * (i + 1))
    return None


def main():
    # 1) пробуем с системными настройками прокси, потом напрямую
    sessions = []
    for trust in (True, False):
        s = requests.Session(); s.trust_env = trust; sessions.append(s)
    session = None
    for s in sessions:
        if get(s, "https://kudago.com/public-api/v1.4/locations/?lang=ru") is not None:
            session = s; break
    if session is None:
        print("\nKudaGo не отвечает. Отключите VPN/прокси (или добавьте kudago.com в прямое подключение) и запустите снова.")
        input("Нажмите Enter для выхода..."); return
    print("Соединение с KudaGo есть. Загружаю...")
    rows, seen = [], set()
    for a, b in PERIODS:
        params = {"location": "spb", "actual_since": ts(a), "actual_until": ts(b, True), "page_size": 100,
                  "fields": "id,title,dates,place,categories", "expand": "place", "text_format": "text"}
        url, first, n = BASE, True, 0
        while url:
            j = get(session, url, params if first else None); first = False
            if not j: print("  страница не получена, пропускаю"); break
            for e in j.get("results", []):
                p = e.get("place") or {}; c = p.get("coords") or {}
                cat = (e.get("categories") or ["other"])[0]
                for d in e.get("dates", []):
                    s, en = d.get("start"), d.get("end")
                    if s is None or s < 0 or s >= FOREVER: continue
                    if en is None or en >= FOREVER: en = s + 3 * 3600
                    # только даты внутри периода (мультидневные события оставляем, если пересекают период)
                    if en < ts(a) or s > ts(b, True): continue
                    key = (e["id"], s)
                    if key in seen: continue
                    seen.add(key)
                    f = lambda t: datetime.fromtimestamp(t, timezone.utc).astimezone().replace(tzinfo=None).isoformat(sep=" ")
                    rows.append(("kudago", e["title"], cat, p.get("title") or "", c.get("lat") or "", c.get("lon") or "", f(s), f(en), ""))
                    n += 1
            url = j.get("next"); time.sleep(0.3)
        print(f"{a} .. {b}: {n} событий")
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["source", "title", "category", "venue", "lat", "lon", "start", "end", "capacity"]); w.writerows(rows)
    print(f"\nГотово: {len(rows)} строк -> {OUT}")
    input("Нажмите Enter для выхода...")


if __name__ == "__main__":
    main()
