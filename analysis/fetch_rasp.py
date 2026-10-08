"""Расписание вокзалов Линии 1 из Яндекс.Расписаний (ключ в переменной RASP_KEY).
Скрипт: python fetch_rasp.py 2026-10-08 7  -> rasp_schedule.csv (дата, вокзал, событие, время, нить, тип)
Ограничение API: от 30 дней назад до 11 месяцев вперёд, 100 записей за запрос.
"""
import os, sys, time
from datetime import date, timedelta
import requests, pandas as pd

KEY = os.environ["RASP_KEY"]; B = "https://api.rasp.yandex.net/v3.0/schedule/"
VOKZALS = {"Финляндский": ("s9602497", "Пл.Ленина"), "Московский": ("s9602494", "Пл. Восстания"),
           "Балтийский": ("s9602498", "Балтийская"), "Витебский": ("s9602496", "Пушкинская")}

def pull(code, d, event):
    rows, off = [], 0
    while True:
        r = requests.get(B, params=dict(apikey=KEY, station=code, date=d, event=event,
                         transport_types="suburban,train", limit=100, offset=off, format="json", lang="ru_RU"), timeout=40)
        if r.status_code != 200:
            print("ошибка", r.status_code, r.text[:120]); break
        j = r.json(); s = j["schedule"]
        for x in s:
            t = x.get(event) or x.get("departure") or x.get("arrival")
            if not t: continue
            th = x["thread"]
            rows.append((d, event, t[:19], th.get("title"), th.get("transport_type"), th.get("express_type") or ""))
        off += len(s)
        if not s or off >= j["pagination"]["total"]: break
        time.sleep(0.2)
    return rows

if __name__ == "__main__":
    d0 = date.fromisoformat(sys.argv[1]); n = int(sys.argv[2]); out = []
    for i in range(n):
        d = (d0 + timedelta(days=i)).isoformat()
        for name, (code, st) in VOKZALS.items():
            for ev in ("arrival", "departure"):
                for r in pull(code, d, ev):
                    out.append((name, st) + r)
        print(d, len(out), flush=True)
    pd.DataFrame(out, columns=["vokzal", "metro_station", "date", "event", "time", "thread", "ttype", "express"]).to_csv("rasp_schedule.csv", index=False)
