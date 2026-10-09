"""Решающие правила и сборка ответа в формате интерфейса (контракт описан в app/static/index.html).

Логика: прогноз входа по станциям на 2 часа -> нагрузка перегонов по направлениям (упрощённая модель корреспонденций) ->
сравнение с ёмкостью плана (пары/час x комфортная вместимость) -> рекомендация по запасу времени и выбору депо.
Все допущения помечены в engine.py (вместимость 120 чел/вагон со слов эксперта, средняя поездка 6 перегонов, калибровка пика 0.85).
"""
import math

import numpy as np
import pandas as pd

import engine as E

DIRS = [{"id": "to_veteranov", "name": "К «Проспекту Ветеранов»"}, {"id": "to_devyatkino", "name": "К «Девяткино»"}]
SECTIONS = {  # (первый перегон, последний перегон) по индексам юг->север; названия концов
    "south": (0, 8, "Проспект Ветеранов", "Площадь Восстания"),
    "north": (9, 17, "Площадь Восстания", "Девяткино"),
}
HOT_EXTRA_MIN = {"devyatkino": 8, "avtovo": 7}   # ход горячего резерва от депо до станции (эксперт: ≈7 мин, возможно 8 до Северного)
DEPOTS = {"devyatkino": ("Депо у станции Девяткино", E.NS - 1), "avtovo": ("Депо у станции Автово", 2)}


def _period(t0, j, hours=1):
    a = t0 + pd.Timedelta(minutes=60 * j); b = a + pd.Timedelta(minutes=60 * hours); return a, b


def _station_reasons(w, c, dev_h, dev_now, month, dow, hour, wxrow):
    out = {}
    typ_fb = w.fb[month]; slot = int(w.slot[c + 1]) if c + 1 < E.T else int(w.slot[c])
    for i in range(E.NS):
        r = []
        if abs(dev_h[i]) >= 0.10:
            r.append({"icon": "🤖", "type": "model", "text": f"Прогноз модели на ближайший час: {dev_h[i]*100:+.0f}% к норме (ориентируется на динамику последних 30 минут, вчера и неделю назад).",
                      "impact_pct": round(float(dev_h[i] * 100)), "source": "модель LightGBM"})
        if abs(dev_now[i]) >= 0.10:
            r.append({"icon": "📈" if dev_now[i] > 0 else "📉", "type": "nowcast", "text": f"Фактический вход за последние 30 минут: {dev_now[i]*100:+.0f}% к норме.",
                      "impact_pct": round(float(dev_now[i] * 100)), "source": "турникеты, 15 мин"})
        if dow == 4 and 15 <= hour <= 19 and not np.isnan(typ_fb[1, slot, i]) and not np.isnan(typ_fb[0, slot, i]) and typ_fb[0, slot, i] > 0:
            fr = typ_fb[1, slot, i] / typ_fb[0, slot, i] - 1
            if abs(fr) >= 0.05:
                r.append({"icon": "📅", "type": "time", "text": "Пятница: вечерний пик начинается раньше, чем в пн–чт.", "impact_pct": round(float(fr * 100)), "source": "история 4 месяцев"})
        if wxrow is not None and wxrow.get("precipitation", 0) >= 0.3 and abs(dev_h[i]) >= 0.10:
            r.append({"icon": "🌧️", "type": "weather", "text": f"Дождь {wxrow['precipitation']*4:.1f} мм/ч: по нашим данным вход в метро при осадках в среднем ниже на 1–3% (эффект малый).",
                      "impact_pct": -2, "source": "метео (Open-Meteo)"})
        out[i] = r
    return out


def build_state(w, c, X_live, day: str, extra=None):
    """c: глобальный индекс текущего (известного) слота. X_live: ряд с данными (возможно с вбросом). Возвращает JSON по контракту интерфейса."""
    month = int(w.month[c]); t_now = E.GRID[c] + pd.Timedelta(minutes=15); hour = t_now.hour; dow = int(w.dow[c])
    Pm = w.forecast(month, X_live, c)                                              # (8,19)
    Nm = np.array([w.norm_adj(c + k) for k in range(1, E.HORIZON + 1)])               # (8,19)
    Nm = np.where(np.isnan(Nm), Pm, Nm)
    # инерция: рост/спад последних 30 минут частично сохраняется; смесь с моделью даёт WAPE 5.42% против 5.86% на ближайший час (проверено на 82 раб. днях)
    a2 = X_live[c - 1:c + 1].sum(0); n2 = np.array([w.norm_adj(c - 1), w.norm_adj(c)]).sum(0); n2 = np.where(np.isnan(n2), a2, n2); dev_now = np.clip(a2 / np.clip(n2, 1, None) - 1, -0.6, 1.0)
    if getattr(w, "ens", None) is None:      # при ансамбле инерция уже входит в него
        wk = np.array([0.5] * 4 + [0.25] * 4)[:, None]; Pm = (1 - wk) * Pm + wk * Nm * (1 + 0.7 * dev_now[None, :])
    we = bool(w.weekend_day[c]); phase = "we" if we else ("am" if hour < 13 else "pm")
    pred_h = [Pm[4 * j:4 * j + 4].sum(0) for j in range(2)]; norm_h = [Nm[4 * j:4 * j + 4].sum(0) for j in range(2)]
    ss = w.share_south[phase]; share = {"to_veteranov": ss, "to_devyatkino": 1 - ss}
    dev_h = pred_h[0] / np.clip(norm_h[0], 1, None) - 1
    wxrow = None
    try:
        wxrow = w.wx.loc[E.GRID[c].floor("15min")].to_dict()
    except KeyError:
        pass

    # ---- рекомендации ----
    recs_raw = []   # по окнам
    no_history = bool(np.isnan(w.NORM[c + 1]).any())   # нет предыдущих дней того же класса за 28 дней: норма из других месяцев, рекомендации не даём
    for j in (range(0) if no_history else range(2)):
        a, b = _period(t_now, j); h_plan = a.hour; P = E.plan_pairs(h_plan, month, we); cap = P * E.CAP_TRAIN
        Lp = {"south": None, "north": None}; Ln = {}
        sp, npd = w.seg_loads(pred_h[j], phase); sn, nn = w.seg_loads(norm_h[j], phase)
        loads = {"to_veteranov": (sp, sn), "to_devyatkino": (npd, nn)}
        for d, (lp, ln) in loads.items():
            for sec, (k0, k1, nm0, nm1) in SECTIONS.items():
                seg = np.arange(k0, k1 + 1); u = lp[seg] / max(cap, 1); dev = lp[seg] / np.clip(ln[seg], 1, None) - 1
                cand = np.where(dev >= E.DEV_UP)[0]
                kmax = int(cand[np.argmax(u[cand])]) if len(cand) else int(np.argmax(u)); um, dm, Lm = float(u[kmax]), float(dev[kmax]), float(lp[seg][kmax])
                frm, to = (nm1, nm0) if d == "to_veteranov" else (nm0, nm1)
                if sec == "north" and d == "to_veteranov": frm, to = "Девяткино", "Площадь Восстания"
                if sec == "north" and d == "to_devyatkino": frm, to = "Площадь Восстания", "Девяткино"
                action, delta = None, 0
                if um >= E.U_HI and dm >= E.DEV_UP:
                    need = math.ceil((Lm / E.U_TARGET - cap) / (E.CAP_TRAIN * 60 / E.LOOP_MIN))
                    dmax = int(math.floor((3600 / E.MIN_INTERVAL_S - P) * E.LOOP_MIN / 60))
                    if dmax < 1:
                        action, delta = "limit", 0          # интервал уже на пределе безопасности: добавлять поезда нельзя
                    else:
                        delta = max(1, min(need, 4, dmax)); action = "up"
                elif dm <= E.DEV_DOWN and float(u.max()) <= E.U_LOW and P - 1 >= 11:
                    action, delta = "down", -1
                if action:
                    recs_raw.append(dict(direction=d, section=sec, j=j, start=a, end=b, action=action, delta=delta, P=P, util=um, dev=dm, frm=frm, to=to, hour=h_plan))
    # слияние окон одного участка и действия
    merged = []
    for r in sorted(recs_raw, key=lambda x: (x["direction"], x["section"], x["j"])):
        m = next((x for x in merged if x["direction"] == r["direction"] and x["section"] == r["section"] and x["action"] == r["action"] and x["j"] + 1 == r["j"]), None)
        if m: m["end"] = r["end"]; m["delta"] = max(m["delta"], r["delta"]) if r["action"] in ("up", "limit") else min(m["delta"], r["delta"]); m["j"] = r["j"]; m["util"] = max(m["util"], r["util"]); m["dev"] = max(m["dev"], r["dev"])
        else: merged.append(dict(r, j0=r["j"]))
    recommendations = []; depot_acts = {"devyatkino": [], "avtovo": []}
    for r in merged:
        P = r["P"]; Pn = P + r["delta"] * 60 / E.LOOP_MIN
        k0, k1, _, _ = SECTIONS[r["section"]]
        names = [(E.ST[i], dev_h[i]) for i in range(E.NS) if (k0 <= i <= k1 + 1)]
        top = [f"{n} {d*100:+.0f}%" for n, d in sorted(names, key=lambda x: -abs(x[1]))[:2] if abs(d) >= 0.05]
        reason = (f"Прогноз нагрузки перегона: {r['util']*100:.0f}% комфортной вместимости, {r['dev']*100:+.0f}% к норме." + (f" Больше всего отклоняются: {', '.join(top)}." if top else "")
                  if r["action"] == "up" else
                  (f"Прогноз нагрузки перегона {r['util']*100:.0f}% ({r['dev']*100:+.0f}% к норме), но интервал уже на пределе ({E.fmt_interval(3600 / P)}, минимум {E.MIN_INTERVAL_S} с): добавлять поезда нельзя. Что можно: сократить оборот на конечных и стоянки, ограничить вход на перегруженных станциях, зонный оборот." + (f" Больше всего отклоняются: {', '.join(top)}." if top else "")
                   if r["action"] == "limit" else f"Поток на {abs(r['dev'])*100:.0f}% ниже нормы, загрузка не выше {E.U_LOW*100:.0f}%: можно снять состав (решает диспетчер)."))
        recommendations.append({"direction": r["direction"], "from": r["frm"], "to": r["to"], "action": r["action"], "trains_delta": int(r["delta"]),
                                "interval_now": E.fmt_interval(3600 / P), "interval_new": E.fmt_interval(3600 / max(Pn, 1e-6)),
                                "period": f"{E.hhmm(r['start'])}–{E.hhmm(r['end'])}", "reason": reason})
        # депо
        if r["action"] == "up":
            depot = "devyatkino" if r["direction"] == "to_veteranov" else "avtovo"; idx0 = {"north": (E.NS - 1 if r["direction"] == "to_veteranov" else E.CENTER), "south": (E.CENTER if r["direction"] == "to_veteranov" else 0)}[r["section"]]
            hops = abs(DEPOTS[depot][1] - idx0); lead = E.READY_MIN + HOT_EXTRA_MIN[depot] + hops * E.HOP_MIN
            cmd = max(r["start"] - pd.Timedelta(minutes=lead), t_now); arrive = cmd + pd.Timedelta(minutes=lead); lag = max(0, int(math.ceil((arrive - r["start"]).total_seconds() / 60)))
            hot, cold = min(r["delta"], 2), max(0, r["delta"] - 2)
            note = f"Состав будет на участке в {E.hhmm(arrive)} (готовность {E.READY_MIN} мин + ход {HOT_EXTRA_MIN[depot]} мин + {hops} перегонов)." + (f" Позже начала периода на {lag} мин: пока сократить интервал и оборот." if lag > 5 else "")
            depot_acts[depot].append({"type": "release", "count": hot, "time": E.hhmm(cmd), "direction": r["direction"], "note": note})
            if cold: depot_acts[depot].append({"type": "release", "count": cold, "time": E.hhmm(cmd + pd.Timedelta(minutes=30)), "direction": r["direction"], "note": "Холодный резерв (≈30 мин на проверку систем) — по решению диспетчера."})
        else:
            depot_acts["avtovo"].append({"type": "return", "count": 1, "time": E.hhmm(r["start"]), "direction": r["direction"], "note": "Спад потока — снять состав с линии в депо «Автово» (решает диспетчер)."})
    depots = [{"id": k, "name": DEPOTS[k][0], "actions": v} for k, v in depot_acts.items()]

    # ---- временной ряд "поездов в час" ----
    timeline = {}
    for d in ("to_veteranov", "to_devyatkino"):
        rows = []
        for i in range(12):
            tt = t_now.floor("h") + pd.Timedelta(hours=i); h = tt.hour; base = E.plan_pairs(h, month, we); rec = base
            for r in merged:
                if r["direction"] == d and r["start"] <= tt + pd.Timedelta(minutes=59) and tt < r["end"]:
                    ad = r["delta"] * 60 / E.LOOP_MIN; ad = math.copysign(max(1, abs(round(ad))), ad)
                    rec = max(rec, base + ad) if r["action"] == "up" else min(rec, base + ad)
            rows.append({"time": f"{h:02d}:00", "base": int(base), "rec": int(round(rec))})
        timeline[d] = rows

    # ---- станции ----
    reasons = _station_reasons(w, c, dev_h, dev_now, month, dow, hour, wxrow); stations = []
    for i in reversed(range(E.NS)):    # от Девяткино к Проспекту Ветеранов
        med = {d: int(round(norm_h[0][i] * share[d][i])) for d in share}; cur = {d: int(round(pred_h[0][i] * share[d][i])) for d in share}
        s = {"id": str(E.NS - i), "name": E.ST[i], "median": med, "current": cur, "reasons": [dict(x, direction=None) for x in reasons[i]]}
        if i in (E.NS - 1, 2): s["depot"] = True
        stations.append(s)

    # ---- факторы города ----
    factors = []
    line_dev = float(pred_h[0].sum() / max(norm_h[0].sum(), 1) - 1)
    if abs(line_dev) >= 0.08:
        factors.append({"icon": "📊", "title": "Отклонение потока по линии", "text": f"Прогноз входа на ближайший час {line_dev*100:+.0f}% к норме для этого времени.", "period": "ближайший час"})
    if 7 <= hour <= 9 or (16 <= hour <= 19): factors.append({"icon": "🏢", "title": "Час пик", "text": "Утренний пик 07:30–09:30 к центру, вечерний 16:30–19:00 от центра.", "period": "сейчас"})
    if dow == 4 and 15 <= hour <= 19: factors.append({"icon": "📅", "title": "Пятница", "text": "Вечерний пик начинается раньше и выше, чем в пн–чт (+13–20% в 15–17 ч).", "period": "15:00–18:00"})
    if wxrow:
        p = wxrow.get("precipitation", 0) * 4; t = wxrow.get("temperature", None)
        if p >= 0.3: factors.append({"icon": "🌧️", "title": "Осадки", "text": f"Дождь/снег {p:.1f} мм/ч. По истории эффект на поток метро малый (−1…−3%).", "period": "сейчас"})
        if t is not None and (t <= -10 or t >= 28): factors.append({"icon": "🌡️", "title": "Экстремальная температура", "text": f"{t:.0f}°C.", "period": "сейчас"})
    # ---- данные для карты и мини-графика ----
    cap0 = max(E.plan_pairs(t_now.hour, month, we) * E.CAP_TRAIN, 1); sp0, np0 = w.seg_loads(pred_h[0], phase); sn0, nn0 = w.seg_loads(norm_h[0], phase)
    segments = {"to_veteranov": [{"a": E.ST[k], "b": E.ST[k + 1], "u": round(float(sp0[k] / cap0), 3), "u_norm": round(float(sn0[k] / cap0), 3)} for k in range(E.NS - 1)],
                "to_devyatkino": [{"a": E.ST[k], "b": E.ST[k + 1], "u": round(float(np0[k] / cap0), 3), "u_norm": round(float(nn0[k] / cap0), 3)} for k in range(E.NS - 1)]}
    past = range(c - 7, c + 1); fut = range(c + 1, c + 1 + E.HORIZON)
    nrm_all = [w.norm_adj(k) for k in list(past) + list(fut)]; nrm_line = [float(np.nansum(x)) if x is not None else None for x in nrm_all]
    series = {"t": [E.hhmm(E.GRID[k]) for k in list(past) + list(fut)], "actual": [round(float(X_live[k].sum())) for k in past] + [None] * E.HORIZON,
              "forecast": [None] * 8 + [round(float(v)) for v in Pm.sum(1)], "norm": [round(v) if v is not None and not math.isnan(v) else None for v in nrm_line], "now_index": 7}
    if no_history:
        factors.append({"icon": "ℹ️", "title": "Мало истории для нормы", "text": "Для этого дня в данных нет предыдущих таких же дней за 4 недели: отклонения показаны к норме из других месяцев, рекомендации отключены.", "period": "день"})
    state = {"updated": t_now.isoformat(), "threshold": E.THRESHOLD_PCT, "directions": DIRS, "factors": factors, "timeline": timeline,
             "recommendations": recommendations, "depots": depots, "stations": stations, "segments": segments, "series": series}
    state["meta"] = {"day": day, "time": E.hhmm(t_now), "model_month_excluded": month, "line_dev_pct": round(line_dev * 100, 1), **(extra or {})}
    return state
