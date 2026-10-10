"""Графики для защиты: (1) прогноз нашего ансамбля против медианы («нормы») и факта; (2) решение на модельном сценарии «День события».

Прогноз: «месяц в проверку» (модели не видели месяц дня), горизонт 30 минут, 19 станций, те же 62 928 строк-слот-станций, что и в README (ансамбль 7,95% против нормы 9,74%).
Медиана = «норма» (M6): медиана до 4 предыдущих дней того же класса за 28 дней. Ансамбль = среднее M1, M3, N2, M5, Chronos-2 (как в app/ensemble.py).
Входные файлы лежат в C:\\metro (ens_all.pkl, ens_oof.pkl; создаются analysis/ensemble_lomo.py, ensemble_all.py).
Запуск из корня репозитория:  python analysis/plot_algorithm_charts.py [forecast|decision|all]
Результат: reports/algo_forecast_vs_median.png, reports/algo_decision_event.png (+ CSV с цифрами рядом)."""
import os
import pickle
import textwrap
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); UP = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "app"))
OUT = os.path.join(ROOT, "reports")

# палитра: синий и оранжевый проверены validate_palette.js (ΔE CVD 24,7, ΔE норм. 33,6, контраст ≥3:1 на #fcfcfb); факт — нейтральный чёрный
BG, INK, INK2, GRID_C = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_OUR, C_MED, C_FACT, C_THR = "#2a78d6", "#eb6834", INK, "#e34948"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG, "axes.edgecolor": "#c9c8c3",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlesize": 11, "axes.labelsize": 9.5,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False})


def style(ax, grid="y"):
    ax.grid(axis=grid, color=GRID_C, lw=0.8); ax.set_axisbelow(True)


def forecast_chart():
    import engine as E
    NS = E.NS
    P = pickle.load(open(os.path.join(UP, "ens_all.pkl"), "rb"))["P"]; o = pickle.load(open(os.path.join(UP, "ens_oof.pkl"), "rb")); y, ev, month = o["y"], o["ev"], o["month"]
    names = ["M1 LGBM-L1", "M3 LGBM-остаток", "N2 MLP-остаток", "M5 инерция", "CC Chronos-2 + норма"]
    m = ev & ~np.isnan(y)
    for k in names + ["M6 норма"]:
        m &= ~np.isnan(P[k])
    ens = np.mean([P[k] for k in names], axis=0); med = P["M6 норма"]
    g = np.arange(len(y)) // NS; ts = E.GRID[g]; svc = (ts - pd.Timedelta(hours=3)).normalize(); hour = ts.hour.values
    df = pd.DataFrame({"g": g[m], "day": svc[m], "hour": hour[m], "month": month[m], "y": y[m], "med": med[m], "ens": ens[m]})
    wape = lambda d, c: (d[c] - d["y"]).abs().sum() / d["y"].sum() * 100
    tot_m, tot_e = wape(df, "med"), wape(df, "ens"); print(f"строк {len(df):,}: медиана {tot_m:.2f}%, ансамбль {tot_e:.2f}%")

    # --- выбор трёх дней по данным (без ручной подгонки): типичный, где выигрыш максимален, где ансамбль хуже всего относительно медианы ---
    df["a_med"] = (df.med - df.y).abs(); df["a_ens"] = (df.ens - df.y).abs(); gb = df.groupby("day")
    dd = pd.DataFrame({"n": gb.size(), "slots": gb.g.nunique(), "med": gb.a_med.sum() / gb.y.sum() * 100, "ens": gb.a_ens.sum() / gb.y.sum() * 100}); dd["dow"] = dd.index.dayofweek
    dd = dd[dd.slots >= 30]; dd["gain"] = dd.med - dd.ens
    cal = pd.read_csv(os.path.join(ROOT, "app", "data", "calendar_2026.csv"), parse_dates=["day"]).set_index("day").code; dd["code"] = cal.reindex(dd.index).fillna(0).values
    reg = dd[(dd.dow < 5) & (dd.code == 0)]            # обычные рабочие дни: на них решает диспетчер; праздники и выходные ведутся вручную
    print(f"дней {len(dd)}: ансамбль лучше медианы в {(dd.gain > 0).sum()}, хуже в {(dd.gain < 0).sum()}")
    typical = reg.assign(dist=lambda d: (d.ens - reg.ens.median()).abs() + (d.med - reg.med.median()).abs()).dist.idxmin()
    best = reg.gain.idxmax(); worst = reg.gain.idxmin()
    picks = [("Типичный рабочий день", typical), ("Рабочий день с наибольшим выигрышем", best), ("Рабочий день с наименьшим выигрышем", worst)]
    for t, d in picks:
        print(t, pd.Timestamp(d).date(), dd.loc[d, ["med", "ens"]].round(1).to_dict())

    fig = plt.figure(figsize=(15, 9.2)); gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.28, left=0.06, right=0.985, top=0.82, bottom=0.08)
    WD = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
    for i, (title, d) in enumerate(picks):
        ax = fig.add_subplot(gs[0, i]); s = df[df.day == d].groupby("g")[["y", "med", "ens"]].sum(); cnt = df[df.day == d].groupby("g").size(); s = s[cnt == NS]
        x = E.GRID[s.index].hour + E.GRID[s.index].minute / 60; x = np.where(x < 3, x + 24, x)
        ax.plot(x, s.y / 1000, color=C_FACT, lw=2, label="факт"); ax.plot(x, s.med / 1000, color=C_MED, lw=2, ls=(0, (5, 2.5)), label="медиана (норма)"); ax.plot(x, s.ens / 1000, color=C_OUR, lw=2, label="наш ансамбль")
        dt = pd.Timestamp(d); ax.set_title(f"{title}\n{dt:%d.%m} ({WD[dt.dayofweek]}), ошибка: медиана {dd.loc[d, 'med']:.1f}% → ансамбль {dd.loc[d, 'ens']:.1f}%", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("время, ч"); ax.set_ylabel("вход по линии за 15 мин, тыс. чел." if i == 0 else ""); ax.set_xlim(6.5, 24.25); ax.set_xticks(range(7, 25, 2)); ax.set_xticklabels([str(h % 24) for h in range(7, 25, 2)]); style(ax)
    h, l = fig.axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.925), fontsize=10.5)
    fig.suptitle("Прогноз на 30 минут вперёд: наш ансамбль против медианы (честная проверка «месяц в проверку»)", fontsize=14, y=0.975, color=INK)

    # --- по месяцам ---
    ax = fig.add_subplot(gs[1, 0]); mm = {2: "фев", 5: "май", 7: "июль", 9: "сен"}; xs = np.arange(4); wd = 0.36
    vm = [wape(df[df.month == k], "med") for k in mm]; ve = [wape(df[df.month == k], "ens") for k in mm]
    b1 = ax.bar(xs - wd / 2 - 0.01, vm, wd, color=C_MED, label="медиана"); b2 = ax.bar(xs + wd / 2 + 0.01, ve, wd, color=C_OUR, label="наш ансамбль")
    for b, v in list(zip(b1, vm)) + list(zip(b2, ve)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels(list(mm.values())); ax.set_ylabel("ошибка WAPE, %"); ax.set_ylim(0, max(vm) * 1.18); style(ax)
    ax.set_title(f"По месяцам: в среднем {tot_m:.2f}% → {tot_e:.2f}%", fontsize=10, color=INK, loc="left"); ax.legend(frameon=False, loc="upper right", fontsize=9)

    # --- по часам ---
    ax = fig.add_subplot(gs[1, 1]); hrs = sorted(df.hour.unique(), key=lambda h: (h < 3, h)); hrs = [h for h in hrs if (df.hour == h).sum() > 300]
    hm = [wape(df[df.hour == h], "med") for h in hrs]; he = [wape(df[df.hour == h], "ens") for h in hrs]; xx = [h + 24 if h < 3 else h for h in hrs]
    ax.fill_between(xx, he, hm, color=C_OUR, alpha=0.10, lw=0); ax.plot(xx, hm, color=C_MED, lw=2, marker="o", ms=4, label="медиана"); ax.plot(xx, he, color=C_OUR, lw=2, marker="o", ms=4, label="наш ансамбль")
    for a, b in ((7, 9.5), (16, 19)):
        ax.axvspan(a, b, color="#8a897f", alpha=0.07, lw=0)
    ax.set_xticks(range(7, 25, 2)); ax.set_xticklabels([str(h % 24) for h in range(7, 25, 2)]); ax.set_xlabel("час суток (серым — пики)"); ax.set_ylabel("ошибка WAPE, %"); style(ax)
    ax.set_title("По часам суток", fontsize=10, color=INK, loc="left"); ax.legend(frameon=False, loc="upper left", fontsize=9)

    # --- какая доля слотов точнее порога ---
    ax = fig.add_subplot(gs[1, 2]); rel = lambda c: (df[c] - df.y).abs() / np.clip(df.y, 1, None) * 100; em, ee = rel("med"), rel("ens"); th = np.linspace(0, 60, 121)
    cm = [(em <= t).mean() * 100 for t in th]; ce = [(ee <= t).mean() * 100 for t in th]
    ax.plot(th, cm, color=C_MED, lw=2, label="медиана"); ax.plot(th, ce, color=C_OUR, lw=2, label="наш ансамбль")
    for t, dy in ((10, -9), (20, -9)):
        a, b = (em <= t).mean() * 100, (ee <= t).mean() * 100; ax.axvline(t, color="#8a897f", lw=0.8, ls=":"); ax.text(t + 1, min(a, b) + dy, f"≤{t}%: {a:.0f}% → {b:.0f}%", fontsize=9, color=INK2)
    ax.set_xlabel("ошибка в слоте, % от факта"); ax.set_ylabel("доля слотов с ошибкой не больше порога, %"); ax.set_ylim(0, 102); style(ax)
    ax.set_title("Какая доля прогнозов точнее порога", fontsize=10, color=INK, loc="left"); ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.savefig(os.path.join(OUT, "algo_forecast_vs_median.png"), dpi=150); plt.close(fig)
    pd.DataFrame({"метрика": ["WAPE медиана", "WAPE ансамбль", "строк"], "значение": [round(tot_m, 2), round(tot_e, 2), len(df)]}).to_csv(os.path.join(OUT, "algo_forecast_vs_median.csv"), index=False, encoding="utf-8")
    print("сохранено: reports/algo_forecast_vs_median.png")


def decision_chart():
    """Решение на модельном сценарии «День события» (как в демо): суббота 12.09 реальных входов + вброс +240% на четырёх северных станциях 12:30–15:00.
    Каждые 15 минут строится состояние (ансамбль, модели не видели месяц дня); диспетчер принимает каждую рекомендацию «добавить составы» так же, как кнопка «Принять» в server.py."""
    import decide as D
    import engine as E
    from ensemble import Ensemble
    day = "2026-09-12"; inj = [{"stations": ["Площадь Мужества", "Политехническая", "Академическая", "Гражданский проспект"], "start": "12:30", "end": "15:00", "pct": 240, "ramp": 45, "decay": 45}]
    W = E.World(); W.ens = Ensemble(W); print("ансамбль:", W.ens.names, "|", W.ens.chronos_note, flush=True)
    base = E.GRID.get_loc(pd.Timestamp(day) + pd.Timedelta(hours=3)); X = W.X.copy()
    for it in inj:                       # вброс как в server.x_live()
        a = base + (int(it["start"][:2]) * 60 + int(it["start"][3:]) - 180) // 15; b = base + (int(it["end"][:2]) * 60 + int(it["end"][3:]) - 180) // 15
        ramp = max(1, int(it["ramp"]) // 15); dec = max(1, int(it["decay"]) // 15); cols = [E.ST.index(n) for n in it["stations"]]
        for k in range(a, b + dec):
            f = min(1.0, (k - a + 1) / ramp) if k < b else max(0.0, 1 - (k - b + 1) / (dec + 1)); X[k, cols] = X[k, cols] * (1 + it["pct"] / 100 * f)
    idx = lambda hm: E.GRID.get_loc(pd.Timestamp(day) + pd.Timedelta(hours=int(hm[:2]), minutes=int(hm[3:]))) - 1
    c0, c1 = idx("11:00"), idx("18:00"); applied, rows, issued = [], [], []
    month = int(W.month[c0]); we = bool(W.weekend_day[c0])
    for c in range(c0, c1 + 1):
        st = D.build_state(W, c, X, day, {}, {"applied": applied, "manual": None}); t = E.GRID[c] + pd.Timedelta(minutes=15)
        segs = st["segments"]; mu = max(s["u"] for d in segs.values() for s in d); mb = max(s["ub"] for d in segs.values() for s in d)
        hour = E.GRID[c + 1].hour; phase = "we" if we else ("am" if hour < 13 else "pm")
        A = X[c + 1:c + 5].sum(0); lp = W.seg_loads(A, phase); P = float(np.mean([E.plan_pairs(E.GRID[c + 1 + k].hour, month, we) for k in range(4)]))
        fact_u = max(lp[0].max(), lp[1].max()) / (P * E.CAP_TRAIN)
        rows.append(dict(t=t, forecast_plan=mb, forecast_acc=mu, fact_plan=fact_u, inflow_fact=float(X[c].sum()), inflow_norm=float(np.nansum(W.norm_adj(c))), f30=float(st["series"]["forecast"][9]) if st["series"]["forecast"][9] is not None else np.nan,
                         int_veter=st["intervals"]["to_veteranov"]["now_s"], int_dev=st["intervals"]["to_devyatkino"]["now_s"], int_plan=st["intervals"]["to_veteranov"]["plan_s"], trains_added=sum(a["trains"] for a in applied)))
        acc = []
        for r in st["recommendations"]:
            if r["action"] in ("up", "limit", "noreserve", "return"):
                issued.append(dict(t=t, action=r["action"], section=r["section"], direction=r["direction"], trains=r["trains_delta"], i_now=r["interval_now_s"], i_new=r["interval_new_s"], depot=r.get("depot"), arrive=r["arrive_hm"],
                                   kinds=",".join(f"{a['count']}{'г' if a.get('kind', 'hot') == 'hot' else 'х'}" for a in r.get("alloc", [])), reason=r["reason"]))
            if r["action"] == "return":          # как server.api_accept: вернуть добавленные составы в депо
                acc.append(("return", r))
            if r["action"] == "up" and r.get("alloc"):
                acc.append(("up", r))
        for kind, r in acc:                       # «Принять» (server.api_accept): после расчёта состояния, действует со следующего шага
            if kind == "return":
                applied = [a for a in applied if a["action"] != "up"]; continue
            bs = {"until": pd.Timestamp(day) + pd.Timedelta(hours=25), "action": "up", "at": c, "interval_before_s": r["interval_now_s"], "interval_after_s": r["interval_new_s"], "route": f"{r['from']} → {r['to']}", "target": r["direction"]}
            applied += [dict(bs, direction=a["direction"], trains=a["count"], arrive=pd.Timestamp(a["enter"]), depot=a["depot"], depot_id=a["depot_id"], kind=a.get("kind", "hot")) for a in r["alloc"]]
    R = pd.DataFrame(rows); I = pd.DataFrame(issued); R.to_csv(os.path.join(OUT, "algo_decision_event.csv"), index=False, encoding="utf-8"); I.to_csv(os.path.join(OUT, "algo_decision_event_recs.csv"), index=False, encoding="utf-8")
    print(I.drop(columns="reason").to_string()) if len(I) else print("рекомендаций нет"); print(R[["t", "forecast_plan", "forecast_acc", "fact_plan", "trains_added", "int_veter", "int_dev"]].round(2).to_string())
    plot_decision(R, I, W.ens.names)
    return R, I


C_ACC = "#1baf7a"     # «с принятыми рекомендациями» (слот 3 палитры; контраст 2,74:1 < 3:1 — поэтому прямые подписи на линиях и CSV рядом)


def plot_decision(R, I, models):
    hx = lambda t: t.hour + t.minute / 60
    x = np.array([hx(t) for t in R.t]); ev0, ev1 = 12.5, 15.75
    fig = plt.figure(figsize=(15, 9.4)); gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.2, left=0.06, right=0.985, top=0.835, bottom=0.125)
    fig.suptitle("Наше решение на модельном сценарии «День события»: суббота 12.09 (реальные входы) + вброс +240% на четырёх северных станциях", fontsize=13.5, y=0.975, color=INK)
    fig.text(0.5, 0.925, "Диспетчер принимает каждую рекомендацию «добавить составы» (как кнопка «Принять» в демо) и «вернуть в депо», когда приток спал.", ha="center", fontsize=10, color=INK2)
    shade = lambda ax: ax.axvspan(ev0, ev1, color="#8a897f", alpha=0.08, lw=0)
    n_hot = 0; n_cold = 0
    for _, r in I[I.action == "up"].iterrows():
        for part in str(r.kinds).split(","):
            if part.endswith("г"): n_hot += int(part[:-1])
            elif part.endswith("х"): n_cold += int(part[:-1])

    # (a) вход по линии: факт, медиана, прогноз на 30 минут
    ax = fig.add_subplot(gs[0, 0]); shade(ax)
    ax.plot(x, R.inflow_fact / 1000, color=C_FACT, lw=2, label="факт (с вбросом)"); ax.plot(x, R.inflow_norm / 1000, color=C_MED, lw=2, ls=(0, (5, 2.5)), label="медиана (норма)")
    ok = R.f30.notna().values; ax.plot(x[ok] + 0.5, R.f30[ok] / 1000, color=C_OUR, lw=2, label="наш прогноз за 30 мин до слота")
    ax.set_xlim(11, 18); ax.set_ylabel("вход по линии за 15 мин, тыс. чел."); ax.set_xlabel("время, ч"); style(ax); ax.legend(frameon=False, fontsize=9, loc="upper right")
    fa = R.inflow_fact.values; ov = [(R.f30.values[i] / fa[i + 2] - 1) for i in range(len(R) - 2) if not np.isnan(R.f30.values[i]) and ev0 <= x[i + 2] <= ev1]; ovm = max(ov) * 100
    ax.set_title(f"1. Входы: медиана не видит всплеск; прогноз ловит его, но в пике завышает до +{ovm:.0f}%", fontsize=10.5, loc="left"); ax.text(ev0 + 0.05, ax.get_ylim()[0] + 0.3, "вброс", fontsize=9, color=INK2)

    # (b) загрузка самого нагруженного перегона
    ax = fig.add_subplot(gs[0, 1]); shade(ax)
    ax.axhline(0.88, color=C_THR, lw=1.1, ls=":"); ax.text(17.95, 0.895, "порог рекомендации 88%", fontsize=9, color=C_THR, ha="right"); ax.axhline(1.0, color="#8a897f", lw=0.9, ls=":"); ax.text(17.95, 1.015, "100% комфортной вместимости", fontsize=9, color=INK2, ha="right")
    ax.plot(x, R.fact_plan, color=C_FACT, lw=1.6, alpha=0.9, label="что реально случилось (по входам, график)")
    ax.plot(x, R.forecast_plan, color=C_OUR, lw=2, ls=(0, (5, 2.5)), label="прогноз при движении по графику")
    ax.plot(x, R.forecast_acc, color=C_ACC, lw=2.6, label="прогноз с принятыми рекомендациями")
    for _, r in I.iterrows():
        tx = hx(r.t); lab = f"+{int(r.trains)}" if r.action == "up" else f"вернуть {abs(int(r.trains))}"
        yv = float(R.loc[R.t == r.t, "forecast_plan"].iloc[0]); ax.plot([tx], [yv], marker="o", ms=7, color=C_OUR, mec=BG, mew=1.5, zorder=5)
        ax.annotate(lab, (tx, yv), xytext=(0, 12) if r.action == "up" else (32, 14), textcoords="offset points", ha="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_xlim(11, 18); ax.set_ylim(0.2, 1.75); ax.set_yticks(np.arange(0.2, 1.41, 0.2)); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); ax.set_ylabel("загрузка перегона в ближайший час"); ax.set_xlabel("время, ч"); style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.set_title("2. Загрузка самого нагруженного перегона: пик ниже, спад быстрее", fontsize=10.5, loc="left")

    # (c) интервал
    ax = fig.add_subplot(gs[1, 0]); shade(ax)
    ax.axhline(113, color=C_THR, lw=1.1, ls=":"); ax.text(17.95, 109, "минимум 113 с (53 состава)", fontsize=9, color=C_THR, ha="right", va="top")
    ax.step(x, R.int_plan, where="post", color="#8a897f", lw=1.8, label="по графику")
    ax.step(x, R.int_veter, where="post", color=C_ACC, lw=2.6, label="с принятым: к «Проспекту Ветеранов»"); ax.step(x, R.int_dev, where="post", color=C_ACC, lw=2.0, ls=(0, (4, 2)), label="с принятым: к «Девяткино»")
    ax.set_xlim(11, 18); ax.set_ylim(95, 200); ax.set_ylabel("интервал между поездами, с (ниже = чаще)"); ax.set_xlabel("время, ч"); style(ax); ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("3. Интервал в секундах — то, чем оперирует диспетчер", fontsize=10.5, loc="left")
    if (R.int_veter < 113).any():
        t0 = x[(R.int_veter < 113).values][0]; ax.annotate("ниже 113 с: плановые пары выросли,\nдобавленные составы остались", (t0 + 0.4, 110), xytext=(t0 + 0.6, 150), fontsize=9, color=C_THR, arrowprops=dict(arrowstyle="-", color=C_THR, lw=0.8))

    # (d) добавленные составы
    ax = fig.add_subplot(gs[1, 1]); shade(ax)
    ax.fill_between(x, R.trains_added, step="post", color=C_ACC, alpha=0.22, lw=0); ax.step(x, R.trains_added, where="post", color=C_ACC, lw=2.6)
    ax.axhline(4, color="#8a897f", lw=1, ls=":"); ax.text(11.1, 4.7, "горячий резерв: 4 состава (2+2)", fontsize=9, color=INK2)
    ax.set_xlim(11, 18); ax.set_ylim(0, max(24, R.trains_added.max() + 3)); ax.set_ylabel("добавлено составов на линии"); ax.set_xlabel("время, ч"); style(ax)
    ax.set_title(f"4. Выпущено {n_hot + n_cold}: {n_hot} горячего резерва + {n_cold} из стоящих в депо (≈30 мин)", fontsize=10.5, loc="left")
    note = (f"Ансамбль прогноза: {', '.join(models)} (без Chronos-2: не хватило видеопамяти). Вброс синтетический (what-if), серым отмечено его время. «Реально случилось» — оценка той же моделью корреспонденций по фактическим входам: "
            "выходов пассажиров в данных нет. Вместимость 960 на состав (120 чел/вагон) — со слов эксперта, не подтверждена.")
    fig.text(0.06, 0.012, "\n".join(textwrap.wrap(note, 185)), fontsize=8.6, color=INK2, va="bottom")
    fig.savefig(os.path.join(OUT, "algo_decision_event.png"), dpi=150); plt.close(fig); print("сохранено: reports/algo_decision_event.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    if what in ("forecast", "all"):
        forecast_chart()
    if what in ("decision", "all"):
        decision_chart()
    if what == "plot":      # только перерисовать по сохранённым CSV (без загрузки моделей)
        plot_decision(pd.read_csv(os.path.join(OUT, "algo_decision_event.csv"), parse_dates=["t"]), pd.read_csv(os.path.join(OUT, "algo_decision_event_recs.csv"), parse_dates=["t"]),
                      ["LightGBM", "LightGBM-остаток", "MLP-остаток", "Инерция"])
