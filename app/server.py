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
from minutes import MinuteFeed  # noqa: E402
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
    {"id": "event", "title": "День события (модель): суббота, у севера линии нарастает приток пассажиров", "day": "2026-09-12", "time": "12:15",
     "inj": [{"stations": ["Площадь Мужества", "Политехническая", "Академическая", "Гражданский проспект"], "start": "12:30", "end": "15:00", "pct": 240, "ramp": 45, "decay": 45}]},
    {"id": "fail", "title": "Сбой на другой линии (модель): утром поток по всей линии нарастает", "day": "2026-05-12", "time": "07:00",
     "inj": [{"stations": list(E.ST), "start": "07:30", "end": "08:45", "pct": 45, "ramp": 45, "decay": 45}]},
    {"id": "feb13", "title": "13 февраля (пт): реальный день, утренний пик у предела", "day": "2026-02-13", "time": "05:45", "inj": []},
    {"id": "calm", "title": "12 мая (вт): обычный вечер, система молчит", "day": "2026-05-12", "time": "16:00", "inj": []},
    {"id": "sat", "title": "12 сентября (сб): реальный всплеск выходного дня", "day": "2026-09-12", "time": "08:30", "inj": []},
    {"id": "hol", "title": "11 мая (пн): праздничный день", "day": "2026-05-11", "time": "10:00", "inj": []},
]
S = {"day": SCENARIOS[0]["day"], "c": 0, "playing": False, "rate": 2.0, "acc": 0.0, "inj": [], "ver": 0, "scenario": SCENARIOS[0]["id"], "applied": [], "manual": None, "m": 0}
CACHE = {}; XL = {"key": None, "X": None}; FD = {"key": None, "feed": None}
FC = {"key": None, "hist": {}}      # прогнозы входа по линии, сделанные в каждую посчитанную минуту (для линии «прогноз, сделанный час назад»)


CAL = pd.read_csv(os.path.join(HERE, "data", "calendar_2026.csv"), parse_dates=["day"]).set_index("day").code


def kind(d):
    d = pd.Timestamp(d); c = int(CAL.get(d, 0))
    return " · выходной" if d.dayofweek >= 5 else (" · праздничный" if c == 1 else (" · сокращённый" if c == 2 else ""))


def day0(day):
    return E.GRID.get_loc(pd.Timestamp(day) + pd.Timedelta(hours=3))


def idx_at(day, hm):
    return E.GRID.get_loc(pd.Timestamp(day) + pd.Timedelta(hours=int(hm[:2]), minutes=int(hm[3:]))) - 1


def minute_at(day, hm):
    """минут от начала GRID до времени hm дня (поминутно)"""
    return int((pd.Timestamp(day) + pd.Timedelta(hours=int(hm[:2]), minutes=int(hm[3:])) - E.GRID[0]) / pd.Timedelta(minutes=1))


def c_off():
    return (S["m"] - 15) // 15, (S["m"] - 15) % 15


def feed():
    key = (S["day"], repr(S["inj"]))
    if FD["key"] != key:
        FD["key"], FD["feed"] = key, MinuteFeed(W, S["day"], S["inj"])
    return FD["feed"]


def x_live():
    key = (S["day"], S["ver"])
    if XL["key"] != key:
        X = W.X.copy(); base = day0(S["day"])
        for it in S["inj"]:
            a = base + (int(it["start"][:2]) * 60 + int(it["start"][3:]) - 180) // 15; b = base + (int(it["end"][:2]) * 60 + int(it["end"][3:]) - 180) // 15
            ramp = max(1, int(it.get("ramp", 0)) // 15); dec = max(1, int(it.get("decay", 0)) // 15) if it.get("decay") else 0
            cols = [E.ST.index(n) for n in it["stations"] if n in E.ST]
            for k in range(a, b + dec):
                f = min(1.0, (k - a + 1) / ramp) if (it.get("ramp") and k < b) else (1.0 if k < b else max(0.0, 1 - (k - b + 1) / (dec + 1)))
                X[k, cols] = X[k, cols] * (1 + float(it["pct"]) / 100 * f)
        XL["key"], XL["X"] = key, X
    return XL["X"]


def set_scenario(sc):
    S.update(day=sc["day"], m=minute_at(sc["day"], sc["time"]), inj=list(sc["inj"]), scenario=sc["id"], ver=S["ver"] + 1, playing=False, acc=0.0, applied=[], manual=None)


def clamp():
    lo, hi = minute_at(S["day"], "05:45"), minute_at(S["day"], "23:00"); S["m"] = max(lo, min(hi, S["m"])); S["c"] = c_off()[0]


set_scenario(SCENARIOS[0])


@app.on_event("startup")
async def _tick():
    async def loop():
        while True:
            await asyncio.sleep(0.25)
            if S["playing"]:
                S["acc"] += 0.25 * S["rate"]          # rate = минут эмулятора в секунду
                while S["acc"] >= 1:
                    S["acc"] -= 1; S["m"] += 1
                if S["m"] >= minute_at(S["day"], "23:00"):
                    S["playing"] = False
                clamp()
    asyncio.create_task(loop())


@app.get("/api/state")
def api_state():
    clamp(); key = (S["day"], S["m"], S["ver"]); c, off = c_off()
    if key not in CACHE:
        fd = feed()
        CACHE.clear(); CACHE[key] = D.build_state(W, c, fd.x_roll(x_live(), c, off), S["day"], {"scenario": S["scenario"], "inj": S["inj"], "playing": S["playing"], "models": W.ens.names, "chronos": W.ens.chronos_note},
                                                  {"applied": S["applied"], "manual": S["manual"]}, off=off, feed=fd)
    fk = (S["day"], repr(S["inj"]))
    if FC["key"] != fk: FC["key"], FC["hist"] = fk, {}
    FC["hist"][S["m"]] = CACHE[key]["series"]["forecast"][8:]
    hour_ago = []
    for i in range(8):                     # окно, закончившееся (7 - i) * 15 минут назад: что о нём говорил прогноз за час до его конца
        m1 = S["m"] - 15 * (7 - i) - 60; got = next((FC["hist"][x] for x in (m1, m1 - 1, m1 + 1, m1 - 2, m1 + 2) if x in FC["hist"]), None)
        hour_ago.append(got[3] if got else None)
    out = dict(CACHE[key]); out["series"] = dict(out["series"], hour_ago=hour_ago + [None] * E.HORIZON); out["meta"] = dict(out["meta"], playing=S["playing"], rate=S["rate"], day=S["day"], dow=WD[pd.Timestamp(S["day"]).dayofweek]); return out


@app.get("/api/emulator")
def api_emulator():
    return {"days": [{"day": str(pd.Timestamp(d).date()), "label": f"{pd.Timestamp(d):%d.%m} ({WD[pd.Timestamp(d).dayofweek]})" + kind(d)} for d in DAYS], "scenarios": [{"id": s["id"], "title": s["title"]} for s in SCENARIOS],
            "day": S["day"], "time": E.hhmm(E.GRID[0] + pd.Timedelta(minutes=S["m"])), "playing": S["playing"], "rate": S["rate"], "inj": S["inj"], "scenario": S["scenario"], "stations": list(reversed(E.ST))}


@app.post("/api/emulator")
def api_emulator_set(body: dict = Body(...)):
    if "scenario" in body:
        sc = next((s for s in SCENARIOS if s["id"] == body["scenario"]), None)
        if sc: set_scenario(sc)
    if "day" in body and body["day"] != S["day"]:
        S.update(day=body["day"], m=minute_at(body["day"], body.get("time", "05:45")), inj=[], scenario="custom", ver=S["ver"] + 1, playing=False, applied=[], manual=None)
    m0 = S["m"]
    if "time" in body: S["m"] = minute_at(S["day"], body["time"])
    if "step" in body: S["m"] += 15 * int(body["step"])
    if "minutes" in body: S["m"] += int(body["minutes"])
    if S["m"] < m0: S["applied"] = []; S["manual"] = None; S["ver"] += 1
    if "playing" in body: S["playing"] = bool(body["playing"])
    if "rate" in body: S["rate"] = float(body["rate"])
    if body.get("clear_inject"): S["inj"] = []; S["ver"] += 1
    if "inject" in body: S["inj"] = S["inj"] + [body["inject"]]; S["ver"] += 1
    clamp(); return api_emulator()


@app.post("/api/accept")
def api_accept(body: dict = Body(...)):
    """Диспетчер принимает рекомендацию (по ключу из /api/state) или отменяет принятое."""
    if body.get("cancel"):
        S["applied"] = []; S["ver"] += 1; return {"ok": True}
    clamp(); api_state()
    st = CACHE[(S["day"], S["m"], S["ver"])]; keys = body.get("keys") or [body.get("key")]
    for r in st["recommendations"]:
        if r["key"] in keys and r["action"] == "return":     # вернуть добавленные составы в депо
            S["applied"] = [a for a in S["applied"] if a["action"] != "up"]
        if r["key"] in keys and r["action"] in ("up", "down"):
            # выпущенный состав остаётся на линии до конца дня, пока диспетчер не снимет
            base = {"until": pd.Timestamp(S["day"]) + pd.Timedelta(hours=25) if r["action"] == "up" else pd.Timestamp(r["until"]), "action": r["action"], "at": S["c"], "interval_before_s": r["interval_now_s"], "interval_after_s": r["interval_new_s"],
                    "route": f"{r['from']} → {r['to']}", "target": r["direction"]}
            if r.get("alloc"):   # по депо: ближнее выходит в нужном направлении, дальнее — в обратном (в нужное через полкруга)
                S["applied"] = list(S["applied"]) + [dict(base, direction=a["direction"], trains=a["count"], arrive=pd.Timestamp(a["enter"]), depot=a["depot"], depot_id=a["depot_id"], kind=a.get("kind", "hot")) for a in r["alloc"]]
            else:
                S["applied"] = list(S["applied"]) + [dict(base, direction=r["direction"], trains=r["trains_delta"], arrive=pd.Timestamp(r["arrive"]), depot=r["depot"], depot_id=None)]
    S["ver"] += 1; return {"ok": True}


@app.post("/api/interval")
def api_interval(body: dict = Body(...)):
    """Текущий интервал, заданный диспетчером вручную (секунды); null — вернуть график."""
    sec = body.get("seconds")
    if sec in (None, 0, ""):
        S["manual"] = None
    else:
        t0 = E.GRID[0] + pd.Timedelta(minutes=S["m"]); S["manual"] = {"from": t0, "to": t0 + pd.Timedelta(hours=3), "interval": max(float(E.MIN_INTERVAL_S), min(600.0, float(sec)))}
    S["applied"] = []; S["ver"] += 1; return {"ok": True}


STATIC = os.path.join(HERE, "static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "dispatcher.html"))


@app.get("/classic")
def classic():
    return FileResponse(os.path.join(STATIC, "classic.html"))
