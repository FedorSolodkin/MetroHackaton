"""Движок демо: данные, прогноз (LightGBM, рекурсивно на 2 часа), нагрузка перегонов по направлениям, решающие правила,
сборка ответа в формате интерфейса «Красная линия: управление интервалами».

Эмулятор проигрывает РЕАЛЬНЫЕ дни. Для дня месяца M используется модель, обученная на остальных трёх месяцах
(честный прогноз вне выборки). Модель видит только данные до текущего момента курсора (лаги >= 30 мин, дальше рекурсия).
"""
import math
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); MODELS = os.path.join(HERE, "models"); APPDATA = os.path.join(HERE, "data")

GRID = pd.date_range("2026-02-01 03:00", "2026-10-01 02:45", freq="15min"); T = len(GRID)
ST = ["Проспект Ветеранов", "Ленинский проспект", "Автово", "Кировский завод", "Нарвская", "Балтийская", "Технологический институт",
      "Пушкинская", "Владимирская", "Площадь Восстания", "Чернышевская", "Площадь Ленина", "Выборгская", "Лесная", "Площадь Мужества",
      "Политехническая", "Академическая", "Гражданский проспект", "Девяткино"]       # с юга на север, индекс 0..18
NS = len(ST); CENTER = 9                                                           # Площадь Восстания
HORIZON = 8                                                                          # слотов вперёд (2 часа)
FEATS = ["st", "slot", "dow", "off", "pre", "off_next", "off_prev", "lag2", "lag3", "lag4", "lag6", "lag8", "lag96", "lag672", "mean_2_5", "trend"]

# --- параметры линии и решений (источники: данные организаторов, слова эксперта; допущения помечены) -------------
CAP_TRAIN = 960            # комфортная вместимость состава: 120 чел/вагон (эксперт; нужно подтвердить)
MIN_INTERVAL_S = 113; LOOP_MIN = 99; HOP_MIN = 2.6; READY_MIN = 5; TO_AVTOVO_MIN = 7
U_HI, U_TARGET, DEV_UP, DEV_DOWN, U_LOW = 0.88, 0.85, 0.08, -0.25, 0.50
THRESHOLD_PCT = 20          # порог аномалии на станциях в интерфейсе
MEAN_TRIP_STOPS = 6.0       # средняя поездка 6 перегонов (коэффициент сменяемости 3)
PLAN_AUTUMN = [7, 20, 30, 32, 26, 22, 20, 18, 18, 20, 24, 28, 30, 30, 28, 20, 15, 14, 11, 7, 0]      # часы 5..23, 0, 1
PLAN_SUMMER = [5, 20, 28, 30, 25, 21, 19, 18, 18, 20, 22, 25, 27, 29, 25, 20, 15, 14, 9, 7, 0]
PLAN_HOURS = list(range(5, 24)) + [0, 1]


def plan_pairs(hour: int, month: int) -> int:
    t = PLAN_SUMMER if month in (6, 7, 8) else PLAN_AUTUMN
    return t[PLAN_HOURS.index(hour)] if hour in PLAN_HOURS else 0


def take(M, idx):
    """M[idx] c NaN вне диапазона. idx: массив индексов."""
    out = np.full(np.shape(idx) + M.shape[1:], np.nan)
    ok = (idx >= 0) & (idx < len(M)); out[ok] = M[idx[ok]]; return out


class World:
    def __init__(self):
        d = pd.read_parquet(os.path.join(DATA, "processed_15min_line1.parquet"), columns=["datetime", "station_name", "passengers"])
        d["datetime"] = pd.to_datetime(d.datetime)
        self.X = d.groupby(["datetime", "station_name"]).passengers.sum().unstack().reindex(GRID)[ST].values.astype(float)   # (T,19)
        cal = pd.read_csv(os.path.join(APPDATA, "calendar_2026.csv"), parse_dates=["day"]).set_index("day")
        svc = (GRID - pd.Timedelta(hours=3)).normalize(); self.svc = svc
        self.slot = (((GRID.hour * 60 + GRID.minute - 180) % 1440) // 15).values.astype(int); self.dow = svc.dayofweek.values
        code = cal.code.reindex(svc).fillna(0).values; self.off = (code == 1).astype(int); self.pre = (code == 2).astype(int)
        offd = cal.code.eq(1).astype(int); self.off_next = offd.shift(-1).reindex(svc).fillna(0).values.astype(int); self.off_prev = offd.shift(1).reindex(svc).fillna(0).values.astype(int)
        self.month = svc.month.values
        self.scope_row = (self.dow < 5) & (self.off == 0) & (self.pre == 0) & (self.slot >= 14) & (self.slot <= 85)
        self.models = {m: lgb.Booster(model_file=os.path.join(MODELS, f"lgb_hold{m}.txt")) for m in (2, 5, 7, 9) if os.path.exists(os.path.join(MODELS, f"lgb_hold{m}.txt"))}
        wx = pd.read_parquet(os.path.join(DATA, "weather_spb_2026.parquet")); wx["datetime"] = pd.to_datetime(wx.datetime); self.wx = wx.set_index("datetime")
        self._medians(); self._od()

    # ---------- нормы ----------
    def _medians(self):
        """Запасная норма: медиана по (тип дня, слот) из ДРУГИХ месяцев, только обычные рабочие дни."""
        self.fb = {}
        typ = np.where(self.dow == 4, 1, 0)
        for m in (2, 5, 7, 9):
            tab = np.full((2, 96, NS), np.nan)
            for t in (0, 1):
                for s in range(14, 86):
                    mk = self.scope_row & (self.month != m) & (typ == t) & (self.slot == s)
                    if mk.any():
                        tab[t, s] = np.nanmedian(self.X[mk], axis=0)
            self.fb[m] = tab
        self.typ = typ

    def norm(self, idx: int) -> np.ndarray:
        """Норма для слота idx: медиана последних 4 таких же дней недели (без праздников), иначе из других месяцев."""
        vals = []
        for k in range(1, 5):
            j = idx - 672 * k
            if j >= 0 and self.off[j] == 0 and self.pre[j] == 0 and not np.isnan(self.X[j]).any():
                vals.append(self.X[j])
        if len(vals) >= 2:
            return np.median(np.array(vals), axis=0)
        return self.fb[int(self.month[idx])][self.typ[idx], self.slot[idx]]

    # ---------- признаки и прогноз ----------
    def feat_rows(self, V, S):
        """Признаки для целевых слотов S (массив) по ряду V. Строки: слот-станция."""
        n = len(S); lag = {k: take(V, S - k) for k in (2, 3, 4, 6, 8)}
        lag96, lag672 = take(self.X, S - 96), take(self.X, S - 672)
        mean = np.nanmean(np.stack([lag[2], lag[3], lag[4]]), axis=0)
        trend = lag[2] - lag[4]
        rep = lambda a: np.repeat(np.asarray(a)[:, None], NS, axis=1)
        cols = {"st": np.tile(np.arange(NS), (n, 1)), "slot": rep(self.slot[S]), "dow": rep(self.dow[S]), "off": rep(self.off[S]), "pre": rep(self.pre[S]),
                "off_next": rep(self.off_next[S]), "off_prev": rep(self.off_prev[S]), "lag2": lag[2], "lag3": lag[3], "lag4": lag[4], "lag6": lag[6], "lag8": lag[8],
                "lag96": lag96, "lag672": lag672, "mean_2_5": mean, "trend": trend}
        return pd.DataFrame({k: cols[k].reshape(-1) for k in FEATS})

    def forecast(self, model_month: int, X_live: np.ndarray, c: int) -> np.ndarray:
        """Рекурсивный прогноз слотов c+1..c+HORIZON для всех станций (по данным до слота c включительно)."""
        V = X_live.copy(); V[c + 1:c + HORIZON + 1] = np.nan; mdl = self.models[model_month]; out = np.zeros((HORIZON, NS))
        saveX = self.X; self.X = X_live
        try:
            for k in range(1, HORIZON + 1):
                s = np.array([c + k]); p = np.clip(mdl.predict(self.feat_rows(V, s)), 0, None); V[c + k] = p; out[k - 1] = p
        finally:
            self.X = saveX
        return out

    # ---------- направления и перегоны ----------
    def _od(self):
        """Простая модель корреспонденций: привлекательность станции-назначения = вечерний (утром) или утренний (днём) вход, затухание по расстоянию."""
        mk = self.scope_row; hour = ((self.slot * 15 + 180) // 60) % 24
        eve = np.nanmean(self.X[mk & (hour >= 16) & (hour <= 19)], axis=0); mor = np.nanmean(self.X[mk & (hour >= 6) & (hour <= 9)], axis=0)
        dist = np.abs(np.arange(NS)[:, None] - np.arange(NS)[None, :]); dec = np.exp(-dist / MEAN_TRIP_STOPS); np.fill_diagonal(dec, 0)
        self.W = {"am": eve[None, :] * dec, "pm": mor[None, :] * dec}
        self.share_south = {p: np.tril(self.W[p], -1).sum(1) / self.W[p].sum(1) for p in self.W}
        # калибровка масштаба: медианный день, пиковая загрузка плана = 0.85 (допущение: график рассчитан на комфортную нагрузку в пик)
        umax = 0
        for h in range(6, 24):
            idx = np.where(self.scope_row & (((self.slot * 15 + 180) // 60) % 24 == h))[0]
            if not len(idx): continue
            e = np.nanmedian(self.X[idx], axis=0) * 4; ph = "am" if h < 13 else "pm"; s, n = self.seg_loads(e, ph)
            umax = max(umax, s.max() / (plan_pairs(h, 9) * CAP_TRAIN), n.max() / (plan_pairs(h, 9) * CAP_TRAIN))
        self.scale = 0.85 / umax if umax else 1.0

    def seg_loads(self, e, phase):
        """Нагрузка перегонов (пасс./час) по направлениям: юг (к Ветеранов) и север (к Девяткино). 18 перегонов."""
        W = self.W[phase]; F = e[:, None] * W / W.sum(1, keepdims=True)
        south = np.array([F[k + 1:, :k + 1].sum() for k in range(NS - 1)]); north = np.array([F[:k + 1, k + 1:].sum() for k in range(NS - 1)])
        return south * getattr(self, "scale", 1.0), north * getattr(self, "scale", 1.0)


def fmt_interval(sec: float) -> str:
    sec = int(round(sec)); m, s = divmod(sec, 60); return f"{m} мин {s} с" if s else f"{m} мин"


def hhmm(ts) -> str:
    return pd.Timestamp(ts).strftime("%H:%M")
