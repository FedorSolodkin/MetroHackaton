"""Крупные мероприятия Санкт-Петербурга со страниц площадок Яндекс.Афиши (живые и архивные копии).

Идея: на странице площадки (концертный зал, стадион, арена) в данных страницы лежит ПОЛНОЕ расписание ее событий
(сеансы с датой и временем) и точные координаты площадки. Это даёт именно крупные события (а не афишу целиком).

Правила: проверяется robots.txt; пауза PAUSE секунд между запросами; при капче / 403 / 429 сбор останавливается
(защиту не обходим). Данные архива берутся из Wayback Machine (публичный архив).

Запуск:
  python afisha_places.py live        - текущее расписание площадок (будущие события, для боя/демо)
  python afisha_places.py archive     - снимки страниц площадок из архива за 2026 (история)
  python afisha_places.py parse       - только разбор уже скачанного кэша
Результат: afisha_places_events.csv (площадка, координаты, событие, дата-время, категория, цена билета)
"""
import json
import os
import re
import sys
import time
import urllib.robotparser as rp

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept-Language": "ru-RU,ru;q=0.9"}
BASE = "https://afisha.yandex.ru"; CACHE = "_afisha_places"; PAUSE = 10.0
SEEDS = ["saint-petersburg/concert/places", "saint-petersburg/concert", "saint-petersburg/sport", "saint-petersburg/show",
         "saint-petersburg/theatre", "saint-petersburg/festival", "saint-petersburg/standup", "saint-petersburg"]
KNOWN = ["/saint-petersburg/sport/places/gazprom-arena", "/saint-petersburg/concert/places/ledovyi-dvorets",
         "/saint-petersburg/concert/places/bkz-oktiabrskii", "/saint-petersburg/concert/places/iubileinyi-spb",
         "/saint-petersburg/concert/places/sevkabel-port"]
S_LIVE = requests.Session(); S_LIVE.trust_env = False


class Stop(Exception):
    pass


def robots():
    r = rp.RobotFileParser(); r.parse(S_LIVE.get(BASE + "/robots.txt", headers=UA, timeout=30).text.splitlines()); return r


def load_state(html):
    m = re.search(r"window\['__APOLLO_STATE__'\]\s*=\s*(\{.*?\});</script>", html, flags=re.S)
    if not m:
        return None
    raw = re.sub(r'new Date\(("[^"]*")\)', r"\1", m.group(1)); raw = re.sub(r"\bundefined\b", "null", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


def parse_place_page(html, snap=None):
    S = load_state(html)
    if not S:
        return None, []
    places = [v for v in S.values() if isinstance(v, dict) and v.get("__typename") == "Place"]
    if len(places) != 1:   # страница одной площадки
        return None, []
    pl = places[0]; co = pl.get("coordinates") or {}
    place = dict(venue=pl.get("title"), place_url=pl.get("url"), address=pl.get("address"),
                 lat=co.get("latitude"), lon=co.get("longitude"))
    rows = []
    for d in walk(S):
        if d.get("__typename") != "PlaceScheduleOtherSessionGroup":
            continue
        evref = (d.get("event") or {}).get("__ref"); ev = S.get(evref, {}); ses = d.get("session") or {}
        if not ev.get("title") or not ses.get("datetime"):
            continue
        pr = (S.get((ses.get("ticket") or {}).get("__ref", ""), {}).get("price") or {})
        cat = re.match(r"/saint-petersburg/([^/]+)/", ev.get("url") or "")
        rows.append(dict(**place, event_id=ev.get("id"), title=ev["title"], category=cat.group(1) if cat else None,
                         datetime=ses["datetime"], hall=ses.get("hallName"), price_min=pr.get("min"), price_max=pr.get("max"),
                         seen_snapshot=snap))
    return place, rows


def place_urls_from(html):
    S = load_state(html)
    if not S:
        return set()
    return {v["url"] for v in S.values() if isinstance(v, dict) and v.get("__typename") in ("Place", "PlacePreview")
            and "/places/" in (v.get("url") or "")}


def get_live(rob, path):
    if not rob.can_fetch("*", BASE + path):
        print("robots.txt запрещает", path); return None
    r = S_LIVE.get(BASE + path, headers=UA, timeout=40)
    if r.status_code in (403, 429) or "showcaptcha" in r.url or "SmartCaptcha" in r.text[:5000]:
        raise Stop(f"{r.status_code}/капча на {path}: останавливаюсь")
    if r.status_code != 200:
        return None
    time.sleep(PAUSE)
    return r.text


def run_live():
    os.makedirs(CACHE, exist_ok=True); rob = robots(); urls = set(KNOWN)
    try:
        for p in SEEDS:
            h = get_live(rob, "/" + p)
            if h:
                found = place_urls_from(h); urls |= found; print(p, "+", len(found), "площадок", flush=True)
        print("площадок к обходу:", len(urls), flush=True)
        for i, u in enumerate(sorted(urls)):
            fn = os.path.join(CACHE, "live_" + u.strip("/").replace("/", "_") + ".html")
            if not os.path.exists(fn):
                h = get_live(rob, u)
                if not h:
                    continue
                open(fn, "w", encoding="utf-8").write(h)
            if i % 10 == 0:
                print("площадка", i, "/", len(urls), flush=True)
    except Stop as e:
        print("СТОП:", e)


def run_archive():
    os.makedirs(CACHE, exist_ok=True)
    urls = sorted(set(KNOWN) | {"/" + re.sub(r"^live_|\.html$", "", f).replace("_", "/", 2) for f in []})
    if os.path.exists("afisha_places_events.csv"):
        urls = sorted(set(urls) | set(pd.read_csv("afisha_places_events.csv").place_url.dropna()))
    for u in urls:
        r = None
        for k in range(3):
            try:
                r = requests.get("https://web.archive.org/cdx/search/cdx", params={"url": "afisha.yandex.ru" + u, "from": "20260101",
                                 "to": "20261007", "output": "json", "filter": "statuscode:200", "collapse": "timestamp:8", "limit": 100},
                                 headers=UA, timeout=90)
                if r.status_code == 200: break
                time.sleep(25)
            except requests.RequestException:
                time.sleep(10)
        snaps = [x[1] for x in r.json()[1:]] if r is not None and r.status_code == 200 else []
        print(u, len(snaps), "снимков", flush=True)
        for ts in snaps:
            fn = os.path.join(CACHE, f"arch_{u.strip('/').replace('/', '_')}_{ts}.html")
            if not os.path.exists(fn):
                try:
                    rr = requests.get(f"https://web.archive.org/web/{ts}id_/https://afisha.yandex.ru{u}", headers=UA, timeout=90)
                    if rr.status_code == 200:
                        open(fn, "w", encoding="utf-8").write(rr.text)
                except requests.RequestException:
                    pass
                time.sleep(8)
        time.sleep(6)


def run_parse():
    rows = []
    for f in sorted(os.listdir(CACHE)):
        m = re.match(r"(live|arch)_.*?_?(\d{14})?\.html$", f)
        snap = re.search(r"_(\d{14})\.html$", f)
        _, r = parse_place_page(open(os.path.join(CACHE, f), encoding="utf-8").read(), snap.group(1) if snap else "live")
        rows += r
    df = pd.DataFrame(rows)
    if df.empty:
        print("событий не найдено"); return
    df = df.sort_values("seen_snapshot").drop_duplicates(["event_id", "datetime", "venue"], keep="first")
    df.to_csv("afisha_places_events.csv", index=False, encoding="utf-8-sig")
    print("сеансов:", len(df), "| событий:", df.event_id.nunique(), "| площадок:", df.venue.nunique(),
          "| диапазон:", df.datetime.min(), "..", df.datetime.max())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "live"
    if mode == "live":
        run_live(); run_parse()
    elif mode == "archive":
        run_archive(); run_parse()
    else:
        run_parse()
