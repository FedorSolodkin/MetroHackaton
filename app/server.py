"""Локальный сервер демо: эмулятор проигрывания реальных дней + прогноз + интерфейс диспетчера.
Запуск (из корня репозитория):  python -m uvicorn app.server:app --port 8000   -> http://localhost:8000
Эмулятор: реальные данные турникетов выбранного дня; модель не видела месяц этого дня при обучении.
Можно вбросить синтетическое событие (what-if): множитель к входу на выбранных станциях и в выбранное время."""
import asyncio
import os
import sys

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import decide as D  # noqa: E402
import engine as E  # noqa: E402

app = FastAPI(title="МетроПульс-1: демо")
W = E.World()
from ensemble import Ensemble  # noqa: E402
W.ens = Ensemble(W)          # ансамбль: LightGBM, LightGBM-остаток, MLP-остаток, инерция (+ Chronos-2, если есть видеокарта)
print('ансамбль:', W.ens.names, '| Chronos:', W.ens.chronos_note, flush=True)
WD = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
valid = (~np.isnan(W.X).any(axis=1)) & np.isin(W.month, list(W.models))
DAYS = [d for d in sorted(set(W.svc[valid])) if (W.svc == d).sum() >= 90]
CENTER = ["Площадь Восстания", "Площадь Ленина", "Чернышевская", "Владимирская"]
SCENARIOS = [
    {"id": "feb13", "title": "13 февраля (пт): штатный утренний пик, система не вмешивается", "day": "2026-02-13", "time": "05:45", "inj": []},
    {"id": "sep11", "title": "11 сентября (пт): штатный утренний пик", "day": "2026-09-11", "time": "05:45", "inj": []},
    {"id": "calm", "title": "12 мая (вт): обычный вечер, система молчит", "day": "2026-05-12", "time": "16:00", "inj": []},
    {"id": "sat", "title": "12 сентября (сб): реальный всплеск выходного дня", "day": "2026-09-12", "time": "08:30", "inj": []},
    {"id": "hol", "title": "11 мая (пн): праздничный день", "day": "2026-05-11", "time": "10:00", "inj": []},
    {"id": "event", "title": "Вброс: +80% у центра 17:15–19:00 (событие или сбой наземного транспорта)", "day": "2026-05-12", "time": "16:00",
     "inj": [{"stations": CENTER, "start": "17:15", "end": "19:00", "pct": 80}]},
    {"id": "fail", "title": "Вброс: +45% по всей линии 07:30–09:00 (сбой на другой линии)", "day": "2026-05-12", "time": "05:45",
     "inj": [{"stations": list(E.ST), "start": "07:30", "end": "09:00", "pct": 45}]},
]
S = {"day": SCENARIOS[0]["day"], "c": 0, "playing": False, "rate": 0.5, "acc": 0.0, "inj": [], "ver": 0, "scenario": SCENARIOS[0]["id"]}
CACHE = {}; XL = {"key": None, "X": None}


CAL = pd.read_csv(os.path.join(HERE, "data", "calendar_2026.csv"), parse_dates=["day"]).set_index("day").code


def kind(d):
    d = pd.Timestamp(d); c = int(CAL.get(d, 0))
    return " · выходной" if d.dayofweek >= 5 else (" · праздничный" if c == 1 else (" · сокращённый" if c == 2 else ""))


def day0(day):
    return E.GRID.get_loc(pd.Timestamp(day) + pd.Timedelta(hours=3))


def idx_at(day, hm):
    return E.GRID.get_loc(pd.Timestamp(day) + pd.Timedelta(hours=int(hm[:2]), minutes=int(hm[3:]))) - 1


def x_live():
    key = (S["day"], S["ver"])
    if XL["key"] != key:
        X = W.X.copy(); base = day0(S["day"])
        for it in S["inj"]:
            a = base + (int(it["start"][:2]) * 60 + int(it["start"][3:]) - 180) // 15; b = base + (int(it["end"][:2]) * 60 + int(it["end"][3:]) - 180) // 15
            cols = [E.ST.index(n) for n in it["stations"] if n in E.ST]
            X[a:b, cols] = X[a:b, cols] * (1 + float(it["pct"]) / 100)
        XL["key"], XL["X"] = key, X
    return XL["X"]


def set_scenario(sc):
    S.update(day=sc["day"], c=idx_at(sc["day"], sc["time"]), inj=list(sc["inj"]), scenario=sc["id"], ver=S["ver"] + 1, playing=False, acc=0.0)


def clamp():
    lo, hi = idx_at(S["day"], "05:45"), idx_at(S["day"], "23:00"); S["c"] = max(lo, min(hi, S["c"]))


set_scenario(SCENARIOS[0])


@app.on_event("startup")
async def _tick():
    async def loop():
        while True:
            await asyncio.sleep(0.25)
            if S["playing"]:
                S["acc"] += 0.25 * S["rate"]
                while S["acc"] >= 1:
                    S["acc"] -= 1; S["c"] += 1
                if S["c"] >= idx_at(S["day"], "23:00"):
                    S["playing"] = False
                clamp()
    asyncio.create_task(loop())


@app.get("/api/state")
def api_state():
    clamp(); key = (S["day"], S["c"], S["ver"])
    if key not in CACHE:
        CACHE.clear(); CACHE[key] = D.build_state(W, S["c"], x_live(), S["day"], {"scenario": S["scenario"], "inj": S["inj"], "playing": S["playing"], "models": W.ens.names, "chronos": W.ens.chronos_note})
    out = dict(CACHE[key]); out["meta"] = dict(out["meta"], playing=S["playing"], rate=S["rate"], day=S["day"], dow=WD[pd.Timestamp(S["day"]).dayofweek]); return out


@app.get("/api/emulator")
def api_emulator():
    return {"days": [{"day": str(pd.Timestamp(d).date()), "label": f"{pd.Timestamp(d):%d.%m} ({WD[pd.Timestamp(d).dayofweek]})" + kind(d)} for d in DAYS], "scenarios": [{"id": s["id"], "title": s["title"]} for s in SCENARIOS],
            "day": S["day"], "time": E.hhmm(E.GRID[S["c"]] + pd.Timedelta(minutes=15)), "playing": S["playing"], "rate": S["rate"], "inj": S["inj"], "scenario": S["scenario"], "stations": list(reversed(E.ST))}


@app.post("/api/emulator")
def api_emulator_set(body: dict = Body(...)):
    if "scenario" in body:
        sc = next((s for s in SCENARIOS if s["id"] == body["scenario"]), None)
        if sc: set_scenario(sc)
    if "day" in body and body["day"] != S["day"]:
        S.update(day=body["day"], c=idx_at(body["day"], body.get("time", "05:45")), inj=[], scenario="custom", ver=S["ver"] + 1, playing=False)
    if "time" in body: S["c"] = idx_at(S["day"], body["time"])
    if "step" in body: S["c"] += int(body["step"])
    if "playing" in body: S["playing"] = bool(body["playing"])
    if "rate" in body: S["rate"] = float(body["rate"])
    if body.get("clear_inject"): S["inj"] = []; S["ver"] += 1
    if "inject" in body: S["inj"] = S["inj"] + [body["inject"]]; S["ver"] += 1
    clamp(); return api_emulator()


STATIC = os.path.join(HERE, "static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "dispatcher.html"))


@app.get("/classic")
def classic():
    return FileResponse(os.path.join(STATIC, "classic.html"))
