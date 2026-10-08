"""Крупные события из архивных копий Яндекс.Афиши (Wayback Machine) за 2026 год.

Как работает: по CDX-индексу архива находим снимки разделов Афиши Санкт-Петербурга, скачиваем сырой HTML
(пауза между запросами, кэш на диске), из встроенного состояния страницы (window['__APOLLO_STATE__'])
достаём события с датами, площадкой и категорией. В снимок попадает только верхняя "карусель" заметных
событий (десятки на страницу), поэтому охват частичный: это именно крупные события.

Запуск:   python wayback_afisha.py            (перечисление + скачивание + разбор)
          python wayback_afisha.py parse      (только разбор уже скачанного)
Результат: afisha_wayback_events.csv
"""
import json
import os
import re
import sys
import time
from datetime import datetime

import pandas as pd
import requests

H = {"User-Agent": "MetroHackaton research (non-commercial student project)"}
CACHE = "_wayback"; PAUSE = 8.0
PAGES = ["saint-petersburg", "saint-petersburg/concert", "saint-petersburg/festival", "saint-petersburg/theatre",
         "saint-petersburg/sport", "saint-petersburg/show", "saint-petersburg/standup", "saint-petersburg/circus_show",
         "saint-petersburg/art", "saint-petersburg/kids"]
FROM, TO = "20260101", "20261007"


def _get(url, params=None, tries=4, timeout=90):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=H, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(25 * (i + 1)); continue
            return None
        except requests.RequestException:
            time.sleep(10 * (i + 1))
    return None


def list_snapshots(path):
    r = _get("https://web.archive.org/cdx/search/cdx", {"url": "afisha.yandex.ru/" + path, "from": FROM, "to": TO,
             "output": "json", "filter": "statuscode:200", "collapse": "timestamp:8", "limit": 300})
    if r is None:
        return []
    j = r.json()
    return [x[1] for x in j[1:]]


def fetch(path, ts):
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, f"{path.replace('/', '_')}_{ts}.html")
    if os.path.exists(fn) and os.path.getsize(fn) > 1000:
        return fn
    r = _get(f"https://web.archive.org/web/{ts}id_/https://afisha.yandex.ru/{path}")
    if r is None:
        return None
    open(fn, "w", encoding="utf-8").write(r.text)
    time.sleep(PAUSE)
    return fn


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


def parse_snapshot(html, snap_ts, page):
    S = load_state(html)
    if not S:
        return []
    rows = []
    for d in walk(S):
        sk = next((k for k in d if str(k).startswith("scheduleInfo")), None)
        if sk is None or not isinstance(d.get(sk), dict):
            continue
        sch = d[sk]; ev_ref = (d.get("event") or {}).get("__ref") if isinstance(d.get("event"), dict) else None
        ev = S.get(ev_ref, {}) if ev_ref else {}
        title = ev.get("title") or d.get("title")
        if not title:
            continue
        typ = S.get((ev.get("type") or {}).get("__ref", ""), {}).get("name") if isinstance(ev.get("type"), dict) else None
        reg = sch.get("regularity") or {}
        place = sch.get("placePreview")
        pref = (sch.get("onlyPlace") or sch.get("oneOfPlaces") or {}).get("__ref") if isinstance(sch.get("onlyPlace") or sch.get("oneOfPlaces"), dict) else None
        purl = S.get(pref, {}).get("url") if pref else None
        dg = [x for k, v in sch.items() if str(k).startswith("dateGroups") and isinstance(v, list) for x in v]
        for dt in (sch.get("dates") or [sch.get("dateStarted")]):
            if not dt:
                continue
            rows.append(dict(snapshot=snap_ts, page=page, event_id=ev.get("id"), title=title, category=typ,
                             url=ev.get("url"), date=dt, date_start=sch.get("dateStarted"), date_end=sch.get("dateEnd"),
                             showtime=reg.get("singleShowtime"), venue=place, place_url=purl,
                             places_total=sch.get("placesTotal"), multi_session=sch.get("multiSession"),
                             permanent=sch.get("permanent"), has_tickets=any(g.get("hasTickets") for g in dg)))
    return rows


def main(parse_only=False):
    jobs = []
    if not parse_only:
        for p in PAGES:
            snaps = list_snapshots(p); print(p, len(snaps), "снимков", flush=True)
            jobs += [(p, ts) for ts in snaps]; time.sleep(6)
        print("всего снимков:", len(jobs), flush=True)
        for i, (p, ts) in enumerate(jobs):
            fetch(p, ts)
            if i % 10 == 0:
                print("скачано", i, "/", len(jobs), flush=True)
    rows = []
    for fn in sorted(os.listdir(CACHE)):
        m = re.match(r"(.+)_(\d{14})\.html$", fn)
        if not m or not fn.startswith("saint-petersburg"):
            continue
        page = m.group(1).replace("_", "/", 1) if m.group(1).startswith("saint-petersburg_") else m.group(1)
        rows += parse_snapshot(open(os.path.join(CACHE, fn), encoding="utf-8").read(), m.group(2), page)
    df = pd.DataFrame(rows)
    if df.empty:
        print("событий не найдено"); return
    df = df.sort_values("snapshot").drop_duplicates(["event_id", "date"], keep="first")
    df.to_csv("afisha_wayback_events.csv", index=False, encoding="utf-8-sig")
    print("событий (уникальных event×date):", len(df), "| событий:", df.event_id.nunique(), "| площадок:", df.venue.nunique())


if __name__ == "__main__":
    main(parse_only=len(sys.argv) > 1 and sys.argv[1] == "parse")
