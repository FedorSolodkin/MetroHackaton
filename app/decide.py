"""Решающие правила и сборка ответа в формате интерфейса (контракт описан в app/static/index.html).

Логика: прогноз входа по станциям на 2 часа -> нагрузка перегонов по направлениям (упрощённая модель корреспонденций) ->
сравнение с ёмкостью плана (пары/час x комфортная вместимость) -> рекомендация по запасу времени и выбору депо.
Все допущения помечены в engine.py (вместимость 120 чел/вагон со слов эксперта, средняя поездка 6 перегонов, калибровка пика 0.85).
"""
import math
import os

import numpy as np
import pandas as pd

import engine as E

DIRS = [{"id": "to_veteranov", "name": "К «Проспекту Ветеранов»"}, {"id": "to_devyatkino", "name": "К «Девяткино»"}]
SECTIONS = {  # (первый перегон, последний перегон) по индексам юг->север; названия концов
    "south": (0, 8, "Проспект Ветеранов", "Площадь Восстания"),
    "north": (9, 17, "Площадь Восстания", "Девяткино"),
}
SURGE_DEV = float(os.environ.get("METRO_SURGE_DEV", 0.25))   # рост входа станции за последние 30 минут, с которого включается поправка «всплеск держится»
SURGE_MIN_ABS = 150   # и при этом не меньше +150 чел. за 30 минут сверх нормы
SURGE_MIN_NORM = 100  # и норма за 30 минут не меньше 100 чел. (иначе это закрытый вестибюль или открытие станции)
HOT_EXTRA_MIN = {"devyatkino": 8, "avtovo": 7}   # ход горячего резерва от депо до станции: Автово 5+7, Северное 5+8 (подтверждено пользователем)
DEPOTS = {"devyatkino": ("Депо «Северное»", E.NS - 1), "avtovo": ("Депо «Автово»", 2)}   # id devyatkino = Северное (у Девяткино)
RESERVE = {"devyatkino": 2, "avtovo": 2}   # горячий резерв: 4 состава, по 2 в каждом депо (данные организаторов)
COLD_READY_MIN = 30                        # составы, стоящие в депо между пиками: бригада + проверка систем ≈30 мин (со слов эксперта «в идеале»; уточнить)
MAX_ADD = 8                                # не больше 8 составов в одной рекомендации
TURN_MIN = 3                               # разворот на конечной
U_BACK = 0.75                              # вернуть добавленные составы, если без них загрузка на 2 часа не выше 75%


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


def build_state(w, c, X_live, day: str, extra=None, ctl=None, off=0, feed=None):
    """c: глобальный индекс текущего (известного) слота. X_live: ряд с данными (возможно с вбросом). Возвращает JSON по контракту интерфейса.
    ctl: состояние диспетчера в эмуляторе: applied (принятые рекомендации: direction, trains, arrive, until, ...) и manual (интервал, заданный вручную: from, to, interval)."""
    ctl = ctl or {}; applied = list(ctl.get("applied", [])); manual = ctl.get("manual")
    # off: минуты после конца слота c (поминутный эмулятор: X_live уже содержит скользящие 15-минутные суммы до этой минуты); feed: поминутный поток для поездов
    month = int(w.month[c]); t_now = E.GRID[c] + pd.Timedelta(minutes=15 + off); hour = t_now.hour; dow = int(w.dow[c])
    Pm = w.forecast(month, X_live, c)                                              # (8,19)
    Nm = np.array([w.norm_adj(c + k) for k in range(1, E.HORIZON + 1)])               # (8,19)
    Nm = np.where(np.isnan(Nm), Pm, Nm)
    # инерция: рост/спад последних 30 минут частично сохраняется; смесь с моделью даёт WAPE 5.42% против 5.86% на ближайший час (проверено на 82 раб. днях)
    a2 = X_live[c - 1:c + 1].sum(0); n2 = np.array([w.norm_adj(c - 1), w.norm_adj(c)]).sum(0); n2 = np.where(np.isnan(n2), a2, n2); dev_now = np.clip(a2 / np.clip(n2, 1, None) - 1, -0.6, 1.0)
    if getattr(w, "ens", None) is None:      # при ансамбле инерция уже входит в него
        wk = np.array([0.5] * 4 + [0.25] * 4)[:, None]; Pm = (1 - wk) * Pm + wk * Nm * (1 + 0.7 * dev_now[None, :])
    # уровень всплеска = отклонение последних 15 минут: на подъёме оно выше среднего за 30 минут, на спаде ниже — прогноз не отстаёт и не переоценивает;
    # держится ближайший час (на спаде 70%; на подъёме плюс часть роста), во втором часу 60% (на спаде 30%)
    dev_raw = np.clip(a2 / np.clip(n2, 1, None) - 1, -0.6, 3.0)                                    # последние 30 минут
    n1 = w.norm_adj(c); n1 = np.where(np.isnan(n1), X_live[c], n1)
    dev15 = np.nan_to_num(np.clip(X_live[c] / np.clip(n1, 1, None) - 1, -0.6, 3.0))                 # последние 15 минут
    surge = (np.maximum(dev_raw, dev15) >= SURGE_DEV) & (n2 >= SURGE_MIN_NORM) & (a2 - n2 >= SURGE_MIN_ABS)   # не срабатывать на закрытых вестибюлях и малых числах
    falling, rising = dev15 < dev_raw - 0.05, dev15 > dev_raw + 0.05
    f1, f2 = np.where(falling, 0.7, 1.0), np.where(falling, 0.3, 0.6)
    ext = np.where(rising, np.minimum(dev15 - dev_raw, 0.25 * dev15), 0.0)                          # рост ещё идёт: немного вперёд, на плато продления нет
    fac = np.where(np.arange(E.HORIZON)[:, None] < 4, 1 + (f1 * dev15 + ext)[None, :], 1 + f2[None, :] * dev15[None, :])
    Pm = np.where(surge[None, :], np.maximum(Pm, Nm * fac), Pm)
    we = bool(w.weekend_day[c]); phase = "we" if we else ("am" if hour < 13 else "pm")
    pred_h = [Pm[4 * j:4 * j + 4].sum(0) for j in range(2)]; norm_h = [Nm[4 * j:4 * j + 4].sum(0) for j in range(2)]
    ss = w.share_south[phase]; share = {"to_veteranov": ss, "to_devyatkino": 1 - ss}
    dev_h = pred_h[0] / np.clip(norm_h[0], 1, None) - 1
    wxrow = None
    try:
        wxrow = w.wx.loc[E.GRID[c].floor("15min")].to_dict()
    except KeyError:
        pass

    # ---- пропускная способность: график (или заданный вручную интервал) и принятые рекомендации ----
    def pairs_base(t):
        """пар в час по графику; если диспетчер задал интервал вручную, то по нему"""
        if manual and manual["from"] <= t < manual["to"]:
            return 3600.0 / manual["interval"]
        return float(E.plan_pairs(t.hour, month, we))

    def boost(d, t):
        """добавленные составы: в своём направлении с момента прибытия на участок, в обратном — через полкруга (доезжают до конца линии и разворачиваются)"""
        sh = pd.Timedelta(minutes=E.LOOP_MIN / 2); tot = 0.0
        for a in applied:
            k = a["trains"] * 60.0 / E.LOOP_MIN
            if a["direction"] == d and a["arrive"] <= t < a["until"]: tot += k
            elif a["direction"] != d and a["arrive"] + sh <= t < a["until"] + sh: tot += k
        return tot

    def pairs(d, t):
        return pairs_base(t) + boost(d, t)

    def pairs_win(d, t0, base_only=False):
        """средняя пропускная способность (пар/час) за час, начинающийся в t0: по четырём слотам, с учётом прибытия добавленных составов внутри часа"""
        ts = [t0 + pd.Timedelta(minutes=15 * k) for k in range(4)]
        return float(np.mean([pairs_base(t) if base_only else pairs(d, t) for t in ts]))

    # ---- поезда (симуляция данных с поездов: положение и загрузка на каждой станции) ----
    tr = None
    if feed is not None:
        import trains as TR
        tr = TR.simulate(w, feed, t_now, pairs, phase, day)
        n_past = np.array([w.norm_adj(c - 1), w.norm_adj(c)]).sum(0); n_past = np.where(np.isnan(n_past), 0, n_past) * 2   # норма последних 30 минут, чел./час
        lnm = dict(zip(("to_veteranov", "to_devyatkino"), w.seg_loads(n_past, phase)))

    # ---- рекомендации ----
    recs_raw = []   # по окнам
    no_history = bool(np.isnan(w.NORM[c + 1]).any())   # нет предыдущих дней того же класса за 28 дней: норма из других месяцев, рекомендации не даём
    for j in (range(0) if no_history else range(2)):
        a, b = _period(t_now, j)
        sp, npd = w.seg_loads(pred_h[j], phase); sn, nn = w.seg_loads(norm_h[j], phase)
        if tr is not None and j == 0 and os.environ.get("METRO_TRAIN_DEC", "1") == "1":          # данные с поездов: перегон за последние 30 минут выше своей нормы на 20% и больше -> то же отклонение на ближайший час
            for d_, lp_, ln_ in (("to_veteranov", sp, sn), ("to_devyatkino", npd, nn)):
                ratio = tr["seg_train_load"][d_] * pairs(d_, t_now) / np.clip(lnm[d_], 1, None)
                lp_[:] = np.where(np.nan_to_num(ratio) >= 1.2, np.fmax(lp_, ln_ * ratio), lp_)
        loads = {"to_veteranov": (sp, sn), "to_devyatkino": (npd, nn)}
        for d, (lp, ln) in loads.items():
            P = pairs_win(d, a); cap = P * E.CAP_TRAIN
            # после прибытия ВСЕХ принятых составов этого направления (дальнее депо приходит через полкруга)
            mine = [x for x in applied if x["action"] == "up" and x.get("target", x["direction"]) == d and x["until"] > a]
            eta = max([x["arrive"] + (pd.Timedelta(0) if x["direction"] == d else pd.Timedelta(minutes=E.LOOP_MIN / 2)) for x in mine], default=None)
            pend = eta is not None and eta > t_now
            P_full = pairs_win(d, a, True) + sum(x["trains"] * 60.0 / E.LOOP_MIN for x in mine); cap_full = P_full * E.CAP_TRAIN
            for sec, (k0, k1, nm0, nm1) in SECTIONS.items():
                seg = np.arange(k0, k1 + 1); u = lp[seg] / max(cap, 1); dev = lp[seg] / np.clip(ln[seg], 1, None) - 1
                cand = np.where(dev >= E.DEV_UP)[0]
                kmax = int(cand[np.argmax(u[cand])]) if len(cand) else int(np.argmax(u)); um, dm, Lm = float(u[kmax]), float(dev[kmax]), float(lp[seg][kmax])
                frm, to = (nm1, nm0) if d == "to_veteranov" else (nm0, nm1)
                if sec == "north" and d == "to_veteranov": frm, to = "Девяткино", "Площадь Восстания"
                if sec == "north" and d == "to_devyatkino": frm, to = "Площадь Восстания", "Девяткино"
                action, delta = None, 0
                extra_r = {}
                if um >= E.U_HI and dm >= E.DEV_UP:
                    need = math.ceil((Lm / E.U_TARGET - cap) / (E.CAP_TRAIN * 60 / E.LOOP_MIN))
                    dmax = int(math.floor((3600 / E.MIN_INTERVAL_S - P) * E.LOOP_MIN / 60))
                    if pend:                     # составы уже едут: досчитываем только то, чего не хватит и после их прибытия
                        need = math.ceil((Lm / E.U_TARGET - cap_full) / (E.CAP_TRAIN * 60 / E.LOOP_MIN))
                        dmax = int(math.floor((3600 / E.MIN_INTERVAL_S - P_full) * E.LOOP_MIN / 60))
                        extra_r = dict(u_full=Lm / cap_full, eta=eta, P_full=P_full)
                    if pend and need <= 0:
                        action, delta = "pending", 0
                    elif dmax < 1:
                        action, delta = "limit", 0          # интервал уже на пределе безопасности: добавлять поезда нельзя
                    else:
                        delta = max(1, min(need, MAX_ADD, dmax)); action = "up"
                elif dm <= E.DEV_DOWN and float(u.max()) <= E.U_LOW and P - 1 >= 11:
                    action, delta = "down", -1
                if action:
                    recs_raw.append(dict(direction=d, section=sec, j=j, start=a, end=b, action=action, delta=delta, P=P, util=um, dev=dm, frm=frm, to=to, hour=a.hour, **extra_r))
    # слияние окон одного участка и действия
    merged = []
    for r in sorted(recs_raw, key=lambda x: (x["direction"], x["section"], x["j"])):
        m = next((x for x in merged if x["direction"] == r["direction"] and x["section"] == r["section"] and x["action"] == r["action"] and x["j"] + 1 == r["j"]), None)
        if m: m["end"] = r["end"]; m["delta"] = max(m["delta"], r["delta"]) if r["action"] in ("up", "limit") else min(m["delta"], r["delta"]); m["j"] = r["j"]; m["util"] = max(m["util"], r["util"]); m["dev"] = max(m["dev"], r["dev"])
        else: merged.append(dict(r, j0=r["j"]))
    recommendations = []; depot_acts = {"devyatkino": [], "avtovo": []}
    used = {k: sum(a["trains"] for a in applied if a.get("depot_id") == k and a["action"] == "up" and a.get("kind", "hot") == "hot") for k in RESERVE}
    left = {k: max(0, RESERVE[k] - used[k]) for k in RESERVE}
    # составы, стоящие в депо между пиками: до 53 минус те, что сейчас на линии по графику (верхняя граница; часть может быть на осмотре)
    on_plan = int(round(E.plan_pairs(t_now.hour, month, we) * E.LOOP_MIN / 60))
    cold_used = sum(a["trains"] for a in applied if a["action"] == "up" and a.get("kind") == "cold")
    cold_left = max(0, E.MAX_TRAINS - on_plan - cold_used); cold_left0 = cold_left
    OPP = {"to_veteranov": "to_devyatkino", "to_devyatkino": "to_veteranov"}
    for r in sorted(merged, key=lambda x: -x["util"]):      # резерв в первую очередь самому нагруженному участку
        r["alloc"], r["short"] = [], 0
        if r["action"] == "up":
            near = "devyatkino" if r["direction"] == "to_veteranov" else "avtovo"; far = "avtovo" if near == "devyatkino" else "devyatkino"
            n1 = min(r["delta"], left[near]); n2 = min(r["delta"] - n1, left[far]); left[near] -= n1; left[far] -= n2
            n3 = min(r["delta"] - n1 - n2, cold_left); cold_left -= n3                  # не хватило горячего резерва — составы из стоящих в депо (≈30 мин)
            r["short"] = r["delta"] - n1 - n2 - n3; r["delta"] = n1 + n2 + n3
            if n1: r["alloc"].append((near, n1, "hot"))
            if n2: r["alloc"].append((far, n2, "hot"))
            if n3: r["alloc"].append((near, n3, "cold"))
            if r["delta"] == 0: r["action"] = "noreserve"
    for r in merged:
        P = r["P"]; Pn = P + r["delta"] * 60 / E.LOOP_MIN
        i_now, i_new = int(round(3600 / P)), int(round(3600 / max(Pn, 1e-6)))
        k0, k1, _, _ = SECTIONS[r["section"]]
        names = [(E.ST[i], dev_h[i]) for i in range(E.NS) if (k0 <= i <= k1 + 1)]
        top = [f"{n} {d*100:+.0f}%" for n, d in sorted(names, key=lambda x: -abs(x[1]))[:2] if abs(d) >= 0.05]
        if r["action"] == "up":
            reason = f"Прогноз нагрузки перегона: {r['util']*100:.0f}% комфортной вместимости, {r['dev']*100:+.0f}% к норме." + (f" Больше всего отклоняются: {', '.join(top)}." if top else "")
        elif r["action"] == "limit":
            reason = (f"Прогноз нагрузки перегона {r['util']*100:.0f}% ({r['dev']*100:+.0f}% к норме), но интервал уже на пределе ({i_now} с, минимум {E.MIN_INTERVAL_S} с): сокращать его нельзя. "
                      "Что можно: сократить оборот на конечных и стоянки, ограничить вход на перегруженных станциях, зонный оборот." + (f" Больше всего отклоняются: {', '.join(top)}." if top else ""))
        elif r["action"] == "pending":
            reason = (f"Принятые составы в пути: последние будут на участке к {E.hhmm(r['eta'])}. Сейчас прогноз нагрузки перегона {r['util']*100:.0f}% вместимости, "
                      f"после их прибытия ≈{r['u_full']*100:.0f}%. Новых действий не требуется.")
        elif r["action"] == "noreserve":
            reason = (f"С учётом уже выпущенных составов прогноз нагрузки перегона {r['util']*100:.0f}% ({r['dev']*100:+.0f}% к норме): не хватает ещё {r['short']} {'состава' if r['short'] == 1 else 'составов' if r['short'] >= 5 else 'составов'}, "
                      "а горячий резерв (по 2 в депо «Северное» и «Автово») и свободные составы в депо уже на линии или в пути. Что можно: сократить оборот на конечных и стоянки, ограничить вход на перегруженных станциях, зонный оборот.")
        else:
            reason = f"Поток на {abs(r['dev'])*100:.0f}% ниже нормы, загрузка не выше {E.U_LOW*100:.0f}%: можно удлинить интервал (решает диспетчер)."
        if r["action"] == "up" and r["short"]:
            reason += f" Нужно было бы ещё {r['short']}, но свободных составов нет: дополнительно сократить оборот на конечных и стоянки."
        rec = {"key": f"{r['direction']}|{r['section']}|{r['action']}", "direction": r["direction"], "section": r["section"], "from": r["frm"], "to": r["to"], "action": r["action"], "trains_delta": int(r["delta"]),
               "interval_now": E.fmt_interval(3600 / P), "interval_new": E.fmt_interval(3600 / max(Pn, 1e-6)),
               "interval_now_s": i_now, "interval_new_s": i_new, "delta_s": i_new - i_now,
               "trains_now": int(round(P * E.LOOP_MIN / 60)), "trains_need": int(round(P * E.LOOP_MIN / 60)) + int(r["delta"]), "trains_max": E.MAX_TRAINS,
               "period": f"{E.hhmm(r['start'])}–{E.hhmm(r['end'])}", "start": E.hhmm(r["start"]), "end": E.hhmm(r["end"]), "reason": reason,
               "until": r["end"].isoformat(), "arrive": r["start"].isoformat(), "arrive_hm": E.hhmm(r["start"]), "depot": None}
        if r["action"] == "pending":
            rec.update(u_now=round(r["util"], 3), u_full=round(r["u_full"], 3), eta=E.hhmm(r["eta"]), interval_new_s=int(round(3600 / r["P_full"])), delta_s=int(round(3600 / r["P_full"])) - i_now)
        recommendations.append(rec)
        # депо: сначала ближнее к участку (выходит сразу в нужном направлении), затем дальнее (выходит в обратном и разворачивается на конечной)
        if r["action"] == "up":
            idx0 = {"north": (E.NS - 1 if r["direction"] == "to_veteranov" else E.CENTER), "south": (E.CENTER if r["direction"] == "to_veteranov" else 0)}[r["section"]]
            alloc = []
            for k, n, kind in r["alloc"]:
                di = DEPOTS[k][1]; ready = E.READY_MIN if kind == "hot" else COLD_READY_MIN; on_line = ready + HOT_EXTRA_MIN[k]
                if (k == "devyatkino") == (r["direction"] == "to_veteranov"):     # ближнее депо
                    lead = on_line + abs(di - idx0) * E.HOP_MIN; enter_dir = r["direction"]
                    how = (f"готовность {ready} мин + ход {HOT_EXTRA_MIN[k]} мин + {abs(di - idx0)} перегонов" if kind == "hot" else
                           f"из стоящих в депо: бригада и проверка ≈{ready} мин + ход {HOT_EXTRA_MIN[k]} мин + {abs(di - idx0)} перегонов; по решению диспетчера")
                else:                                                                # дальнее депо: до конечной, разворот, обратно до участка
                    end = E.NS - 1 if k == "avtovo" else 0; lead = on_line + (abs(end - di) + abs(end - idx0)) * E.HOP_MIN + TURN_MIN; enter_dir = OPP[r["direction"]]
                    how = f"выходит в обратном направлении, разворот на конечной; по пути усиливает обратное направление"
                cmd = max(r["start"] - pd.Timedelta(minutes=lead), t_now); arrive = cmd + pd.Timedelta(minutes=lead); enter = cmd + pd.Timedelta(minutes=on_line)
                lag = max(0, int(math.ceil((arrive - r["start"]).total_seconds() / 60)))
                note = f"На участке в {E.hhmm(arrive)} ({how})." + (f" Позже начала периода на {lag} мин: пока сократить оборот на конечных." if lag > 5 else "")
                depot_acts[k].append({"type": "release", "count": n, "time": E.hhmm(cmd), "direction": r["direction"], "key": rec["key"], "note": note, "cold": kind == "cold"})
                alloc.append({"depot_id": k, "depot": DEPOTS[k][0], "count": n, "direction": enter_dir, "enter": (enter if enter_dir != r["direction"] else arrive).isoformat(), "arrive_hm": E.hhmm(arrive), "kind": kind})
            first = min(alloc, key=lambda x: x["arrive_hm"]) if alloc else None
            rec.update(alloc=alloc, arrive=first["enter"] if first else rec["arrive"], arrive_hm=first["arrive_hm"] if first else rec["arrive_hm"], depot=" и ".join(a["depot"] for a in alloc), short=int(r["short"]))
        elif r["action"] == "down":
            depot_acts["avtovo"].append({"type": "return", "count": 1, "time": E.hhmm(r["start"]), "direction": r["direction"], "key": rec["key"], "note": "Спад потока — можно снять состав с линии в депо «Автово» (решает диспетчер)."})
            rec.update(depot=DEPOTS["avtovo"][0])
    # всплеск закончился: добавленные составы можно вернуть в депо (резерв восстановится), если без них загрузка на 2 часа не выше U_BACK
    n_extra = sum(a["trains"] for a in applied if a["action"] == "up" and a["arrive"] <= t_now)
    if n_extra and not any(r["action"] in ("up", "noreserve", "limit", "pending") for r in recommendations):
        ub_max = 0.0
        for j in range(2):
            a0, _ = _period(t_now, j); capb = max(pairs_win("to_veteranov", a0, True) * E.CAP_TRAIN, 1); lp = w.seg_loads(pred_h[j], phase)
            ub_max = max(ub_max, float(max(lp[0].max(), lp[1].max()) / capb))
        if ub_max <= U_BACK:
            i_now = int(round(3600 / max(pairs_win("to_veteranov", t_now), 1e-6))); i_new = int(round(3600 / max(pairs_win("to_veteranov", t_now, True), 1e-6)))
            recommendations.append({"key": "line|all|return", "direction": "to_veteranov", "section": "all", "from": "Девяткино", "to": "Проспект Ветеранов", "action": "return", "trains_delta": -int(n_extra),
                                    "interval_now": E.fmt_interval(i_now), "interval_new": E.fmt_interval(i_new), "interval_now_s": i_now, "interval_new_s": i_new, "delta_s": i_new - i_now,
                                    "trains_now": int(round(pairs_win("to_veteranov", t_now) * E.LOOP_MIN / 60)), "trains_need": int(round(pairs_win("to_veteranov", t_now, True) * E.LOOP_MIN / 60)), "trains_max": E.MAX_TRAINS,
                                    "period": f"{E.hhmm(t_now)}–{E.hhmm(t_now + pd.Timedelta(hours=2))}", "start": E.hhmm(t_now), "end": E.hhmm(t_now + pd.Timedelta(hours=2)),
                                    "reason": f"Приток спал: прогноз загрузки перегонов на 2 часа без добавленных составов не выше {ub_max*100:.0f}%. Можно вернуть {n_extra} {'состав' if n_extra == 1 else 'состава' if n_extra < 5 else 'составов'} в депо — горячий резерв восстановится.",
                                    "until": (t_now + pd.Timedelta(hours=2)).isoformat(), "arrive": t_now.isoformat(), "arrive_hm": E.hhmm(t_now), "depot": "депо «Северное» и «Автово»"})
    depots = [{"id": k, "name": DEPOTS[k][0], "actions": v} for k, v in depot_acts.items()]

    # ---- временной ряд "поездов в час" ----
    timeline = {}
    for d in ("to_veteranov", "to_devyatkino"):
        rows = []
        for i in range(12):
            tt = t_now.floor("h") + pd.Timedelta(hours=i); h = tt.hour; base = round(pairs_base(tt + pd.Timedelta(minutes=30))); rec_p = base
            for r in merged:
                if r["direction"] == d and r["start"] <= tt + pd.Timedelta(minutes=59) and tt < r["end"]:
                    ad = r["delta"] * 60 / E.LOOP_MIN; ad = math.copysign(max(1, abs(round(ad))), ad)
                    rec_p = max(rec_p, base + ad) if r["action"] == "up" else min(rec_p, base + ad)
            rows.append({"time": f"{h:02d}:00", "base": int(base), "rec": int(round(rec_p))})
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
    # ---- данные для карты и мини-графиков ----
    sp0, np0 = w.seg_loads(pred_h[0], phase); sn0, nn0 = w.seg_loads(norm_h[0], phase)
    cap_a = {d: max(pairs_win(d, t_now) * E.CAP_TRAIN, 1) for d in ("to_veteranov", "to_devyatkino")}; cap_b = max(pairs_win("to_veteranov", t_now, True) * E.CAP_TRAIN, 1)
    segments = {"to_veteranov": [{"a": E.ST[k], "b": E.ST[k + 1], "u": round(float(sp0[k] / cap_a["to_veteranov"]), 3), "ub": round(float(sp0[k] / cap_b), 3), "u_norm": round(float(sn0[k] / cap_b), 3)} for k in range(E.NS - 1)],
                "to_devyatkino": [{"a": E.ST[k], "b": E.ST[k + 1], "u": round(float(np0[k] / cap_a["to_devyatkino"]), 3), "ub": round(float(np0[k] / cap_b), 3), "u_norm": round(float(nn0[k] / cap_b), 3)} for k in range(E.NS - 1)]}
    past = range(c - 7, c + 1); fut = range(c + 1, c + 1 + E.HORIZON)
    nrm_all = [w.norm_adj(k) for k in list(past) + list(fut)]; nrm_line = [float(np.nansum(x)) if x is not None else None for x in nrm_all]
    series = {"t": [E.hhmm(E.GRID[k] + pd.Timedelta(minutes=off)) for k in list(past) + list(fut)], "actual": [round(float(X_live[k].sum())) for k in past] + [None] * E.HORIZON,
              "forecast": [None] * 8 + [round(float(v)) for v in Pm.sum(1)], "norm": [round(v) if v is not None and not math.isnan(v) else None for v in nrm_line], "now_index": 7}
    # загрузка самого нагруженного перегона за час, центрированный на слоте (факт до «сейчас», прогноз после): на слоте «сейчас+45 мин» совпадает с оценкой, по которой принимается решение;
    # «до» = график или заданный вручную интервал, «после» = с принятыми рекомендациями
    flows = [X_live[k] for k in past] + [Pm[i] for i in range(E.HORIZON)]
    SL = [w.seg_loads(f, phase) for f in flows]; ub, ua = [None] * 2, [None] * 2
    for n in range(2, len(flows) - 1):
        ts = E.GRID[c - 7 + n - 2] + pd.Timedelta(minutes=off); rs = sum(SL[m][0] for m in range(n - 2, n + 2)), sum(SL[m][1] for m in range(n - 2, n + 2))
        cb = max(pairs_win("to_veteranov", ts, True) * E.CAP_TRAIN, 1)
        ub.append(round(float(max(rs[0].max(), rs[1].max()) / cb), 3))
        ua.append(round(float(max(rs[0].max() / max(pairs_win("to_veteranov", ts) * E.CAP_TRAIN, 1), rs[1].max() / max(pairs_win("to_devyatkino", ts) * E.CAP_TRAIN, 1))), 3))
    ub.append(None); ua.append(None)
    util_series = {"t": series["t"], "before": ub, "after": ua, "now_index": 7, "u_hi": E.U_HI}
    plan_now = E.plan_pairs(t_now.hour, month, we)
    intervals = {d: {"plan_s": int(round(3600 / plan_now)) if plan_now else None, "now_s": int(round(3600 / max(pairs(d, t_now), 1e-6))),
                     "manual": bool(manual and manual["from"] <= t_now < manual["to"])} for d in ("to_veteranov", "to_devyatkino")}
    act = [{"direction": a["direction"], "trains": a["trains"], "from": E.hhmm(a["arrive"]), "until": E.hhmm(a["until"]), "interval_before_s": a["interval_before_s"], "interval_after_s": a["interval_after_s"],
            "action": a["action"], "route": a["route"], "depot": a.get("depot"), "target": a.get("target", a["direction"])} for a in applied if a["until"] > t_now]
    if no_history:
        factors.append({"icon": "ℹ️", "title": "Мало истории для нормы", "text": "Для этого дня в данных нет предыдущих таких же дней за 4 недели: отклонения показаны к норме из других месяцев, рекомендации отключены.", "period": "день"})
    state = {"updated": t_now.isoformat(), "threshold": E.THRESHOLD_PCT, "directions": DIRS, "factors": factors, "timeline": timeline,
             "recommendations": recommendations, "depots": depots, "stations": stations, "segments": segments, "series": series,
             "util_series": util_series, "intervals": intervals, "applied": act, "reserve_left": int(sum(max(0, RESERVE[k] - used[k]) for k in RESERVE)), "cold_left": int(cold_left0)}
    if tr is not None:
        for d_ in ("to_veteranov", "to_devyatkino"):
            for k_, sg in enumerate(segments[d_]):
                v = tr["seg_train_load"][d_][k_]; sg["um"] = None if np.isnan(v) else round(float(v / E.CAP_TRAIN), 3)
        state["trains"] = tr["trains"]
        state["line_trains"] = int(round(np.mean([pairs(d_, t_now) for d_ in ("to_veteranov", "to_devyatkino")]) * E.LOOP_MIN / 60))   # всего на линии, с разворотами на конечных
    state["meta"] = {"day": day, "time": E.hhmm(t_now), "model_month_excluded": month, "line_dev_pct": round(line_dev * 100, 1), **(extra or {})}
    return state
