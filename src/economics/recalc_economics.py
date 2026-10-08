"""Пересчёт экономического эффекта адаптивного насыщения Линии 1 (снятие избыточных пар в спад спроса).

Старый расчёт (eda_summary_metrics.json, отчёт, разд. 6) использовал 196 кВт·ч/поезд-км (24.5 x 8 вагонов,
хотя 24.5 - это уже на поезд), 120 "оптимизированных кругов" без связи со сценарием и называл итог за 120 суток
"в месяц". Здесь расчёт строится из данных и явных допущений, а результат разделён на:
  - прямую экономию метрополитена (электроэнергия + ТОиР),
  - эффект для пассажиров (рост ожидания) - показывается отдельно, чтобы не выдавать его за чистую выгоду.

Запуск: python -m src.economics.recalc_economics
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.metro_config import METRO_CONFIG, ENERGY_CONFIG, COST_CONFIG
from src.data.operating_hours import filter_operating_hours
from src.data.data_loader import MetroDataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- допущения (все помечены; заменить реальными данными метрополитена) -----------------------
P = dict(
    round_km=59.2,                                  # полный оборот (длина линии 29.6 км x 2)
    kwh_per_train_km=ENERGY_CONFIG["specific_kwh_per_train_km"],   # 24.5 из конфига, НУЖНО подтвердить у метро
    tariff=ENERGY_CONFIG["tariff_rub_per_kwh"],     # 6.8 руб/кВт·ч, допущение
    toir_rub_per_km=COST_CONFIG["toir_cost_per_km"],  # 120 руб/км, допущение
    capacity=1460,                                  # комфортная вместимость состава (5 чел/м2)
    util_peak=0.85,                                 # допущение для калибровки: на пике самый нагруженный перегон ~85% комфорта
    p_min=11,                                       # мин. парность: макс. интервал по графику 5:30 -> ~11 пар/час
    hops=18,                                        # перегонов на линии
    trip_hops=18 / METRO_CONFIG["passenger_turnover_ratio"],      # средняя поездка = 18/3 = 6 перегонов
    safe_util=0.60,                                 # снимаем пары только в часы, где загрузка плана < 60%
    value_of_time_rub_h=300.0,                      # ценность часа пассажира, допущение
    weekdays=22, weekend_days=8,                    # сентябрь 2026
    adoption=0.25,                                  # доля теоретического потенциала, реально принимаемая диспетчером
)


def hourly_entries(month: int = 9):
    df = filter_operating_hours(pd.read_parquet(os.path.join(ROOT, "data", "processed_15min_line1.parquet")))
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[df.datetime.dt.month == month]
    df["day"] = (df.datetime - pd.Timedelta(hours=3)).dt.normalize()
    df["hour"] = df.datetime.dt.hour
    hh = df.groupby(["day", "hour"]).passengers.sum().reset_index()
    hh["weekend"] = hh.day.dt.dayofweek >= 5
    return {w: hh[hh.weekend == w].groupby("hour").passengers.median() for w in (False, True)}


def scenario(E, sched, p, k):
    """Снятие пар в часы низкой загрузки. Возвращает таблицу по часам."""
    rows = []
    for _, r in sched.iterrows():
        h, plan = int(r.hour), int(r.pairs_per_hour)
        e = float(E.get(h, 0.0))
        util_plan = k * e * p["trip_hops"] / (2 * p["hops"] * max(plan, 1) * p["capacity"]) if plan else 0.0
        req = math.ceil(k * e * p["trip_hops"] / (2 * p["hops"] * 0.8 * p["capacity"])) if e else 0
        new = plan
        if plan > p["p_min"] and util_plan < p["safe_util"]:
            new = min(plan, max(req, p["p_min"]))
        d = plan - new
        wait_extra_s = (3600 / new - 3600 / plan) / 2 if (d > 0 and new) else 0.0
        rows.append(dict(hour=h, plan=plan, entries=e, util_plan=util_plan, new=new, removed=d,
                         pax_hours_lost=e * wait_extra_s / 3600))
    return pd.DataFrame(rows)


def money(km, p):
    kwh = km * p["kwh_per_train_km"]
    return dict(km=km, kwh=kwh, energy_rub=kwh * p["tariff"], toir_rub=km * p["toir_rub_per_km"],
                total_rub=kwh * p["tariff"] + km * p["toir_rub_per_km"])


def main():
    p = dict(P)
    E = hourly_entries()
    prof = MetroDataLoader().get_planned_schedule_profiles()
    wd, we = prof["workday_autumn"], prof["weekend"]
    # калибровка k: на пиковом часе рабочего дня plan должен давать util_peak
    ph = wd.loc[wd.pairs_per_hour.idxmax()]
    e_peak = float(E[False].get(int(ph.hour)))
    k = p["util_peak"] * 2 * p["hops"] * ph.pairs_per_hour * p["capacity"] / (e_peak * p["trip_hops"])
    out = {"assumptions": {**p, "k_peak_concentration": k, "peak_hour": int(ph.hour), "peak_entries_per_hour": e_peak}}

    res = {}
    for name, sch, E_, days in (("workday", wd, E[False], p["weekdays"]), ("weekend", we, E[True], p["weekend_days"])):
        t = scenario(E_, sch, p, k)
        daily_km = float(t.removed.sum() * p["round_km"])
        res[name] = dict(table=t.round(3).to_dict("records"), pairs_removed_per_day=int(t.removed.sum()),
                         km_per_day=daily_km, pax_hours_lost_per_day=float(t.pax_hours_lost.sum()),
                         total_plan_pairs=int(t.plan.sum()), days_per_month=days)
    km_month = sum(v["km_per_day"] * v["days_per_month"] for v in res.values())
    plan_km_month = sum(v["total_plan_pairs"] * p["round_km"] * v["days_per_month"] for v in res.values())
    cash = money(km_month, p)
    pax_h = sum(v["pax_hours_lost_per_day"] * v["days_per_month"] for v in res.values())
    out["adaptive_offpeak"] = dict(
        km_month=km_month, share_of_plan_km=km_month / plan_km_month, **{k_: v for k_, v in cash.items() if k_ != "km"},
        passenger_hours_lost_month=pax_h, passenger_time_cost_rub_month=pax_h * p["value_of_time_rub_h"],
        net_after_passenger_time_rub_month=cash["total_rub"] - pax_h * p["value_of_time_rub_h"],
        workday=res["workday"], weekend=res["weekend"])
    ad = p["adoption"]
    r_cash = money(km_month * ad, p)
    out["realistic"] = dict(adoption=ad, km_month=km_month * ad, kwh_month=r_cash["kwh"], energy_rub=r_cash["energy_rub"],
                            toir_rub=r_cash["toir_rub"], total_rub_month=r_cash["total_rub"], total_rub_year=r_cash["total_rub"] * 12,
                            passenger_time_cost_rub_month=pax_h * ad * p["value_of_time_rub_h"],
                            net_rub_month=r_cash["total_rub"] - pax_h * ad * p["value_of_time_rub_h"])

    # чувствительность к удельному расходу энергии и допущению калибровки
    sens = []
    for kwh in (12.0, 24.5):
        for up in (0.70, 0.85, 1.0):
            pp = {**p, "kwh_per_train_km": kwh, "util_peak": up}
            kk = up * 2 * p["hops"] * ph.pairs_per_hour * p["capacity"] / (e_peak * p["trip_hops"])
            km = sum(scenario(E_, sch, pp, kk).removed.sum() * p["round_km"] * d for E_, sch, d in
                     ((E[False], wd, p["weekdays"]), (E[True], we, p["weekend_days"])))
            sens.append(dict(kwh_per_km=kwh, util_peak=up, km_month=km, total_rub_month=money(km, pp)["total_rub"]))
    out["sensitivity"] = sens

    # исходный сценарий напарника, пересчитанный правильно: 120 кругов за 120 суток
    old = money(120 * p["round_km"], p)
    out["old_claim_recomputed"] = dict(trips=120, period_days=120, **old,
                                       per_month_30d=old["total_rub"] / 4, per_year=old["total_rub"] * 3,
                                       claimed_per_month_rub=10_320_691.2, claimed_per_year_rub=123_848_294.4)
    # зонный оборот: экономия на 1 рейсе = полный оборот минус короткий
    short_km = 2 * 9 * p["round_km"] / 2 / 18 * 1  # 9 перегонов x2 стороны
    save_trip_km = p["round_km"] - 2 * 9 * (p["round_km"] / 2 / 18)
    out["short_turn"] = dict(saved_km_per_trip=save_trip_km,
                             per_10_trips_per_day_month=money(10 * 30 * save_trip_km, p))
    path = os.path.join(ROOT, "eda", "economics_recalc.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(json.dumps({k_: out[k_] for k_ in ("assumptions",)}, ensure_ascii=False, indent=1, default=float)[:600])
    a = out["adaptive_offpeak"]
    print(f"\nСнято пар в будни/день: {res['workday']['pairs_removed_per_day']}, в выходные: {res['weekend']['pairs_removed_per_day']}")
    print(f"Пробег: {a['km_month']:,.0f} км/мес ({a['share_of_plan_km']:.1%} планового); энергия {a['energy_rub']:,.0f} руб; ТОиР {a['toir_rub']:,.0f} руб; итого {a['total_rub']:,.0f} руб/мес")
    print(f"Потери времени пассажиров: {a['passenger_hours_lost_month']:,.0f} чел-ч/мес = {a['passenger_time_cost_rub_month']:,.0f} руб (ценность часа {p['value_of_time_rub_h']:.0f})")
    print("Старый сценарий (120 кругов) верно:", {k_: round(v) for k_, v in out["old_claim_recomputed"].items() if k_ in ("km", "kwh", "energy_rub", "toir_rub", "total_rub", "per_month_30d", "per_year")})
    print("Чувствительность (руб/мес):", [(s["kwh_per_km"], s["util_peak"], round(s["total_rub_month"])) for s in sens])
    print("зонный оборот: экономия км на рейс", round(save_trip_km, 1))
    return out


if __name__ == "__main__":
    main()
