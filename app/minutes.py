"""Поминутный поток входа для эмулятора (модель). В данных есть только 15-минутные суммы: минуты получаем плавной интерполяцией
интенсивности с сохранением суммы каждого 15-минутного слота. Модели работают с прежними 15-минутными суммами, но скользящим окном,
которое заканчивается на текущей минуте, — так система обновляется каждую минуту (в реальной системе: поминутный поток турникетов)."""
import numpy as np
import pandas as pd

import engine as E


def _hm(day, hm):
    """'HH:MM' служебных суток (с 03:00) -> Timestamp"""
    h, m = int(hm[:2]), int(hm[3:]); t = pd.Timestamp(day) + pd.Timedelta(hours=h, minutes=m)
    return t + pd.Timedelta(days=1) if h < 3 else t


class MinuteFeed:
    def __init__(self, w, day, inj):
        base = E.GRID.get_loc(pd.Timestamp(day) + pd.Timedelta(hours=3))
        s0 = max(0, base - 8 * 96); s1 = min(E.T, base + 96)                  # неделя назад нужна для лагов и Chronos-2
        X = w.X[s0:s1]; n = len(X); self.s0, self.t0 = s0, E.GRID[s0]
        r = np.nan_to_num(X) / 15.0; cen = np.arange(n) + 0.5; tm = (np.arange(n * 15) + 0.5) / 15.0
        M = np.stack([np.interp(tm, cen, r[:, j]) for j in range(E.NS)], 1).reshape(n, 15, E.NS)
        tot = M.sum(1, keepdims=True); Xz = np.nan_to_num(X)[:, None, :]
        M = np.where(tot > 0, M / np.where(tot > 0, tot, 1) * Xz, 0.0)        # сумма по слоту = исходные 15 минут
        M = np.where(np.isnan(X)[:, None, :], np.nan, M).reshape(n * 15, E.NS)
        for it in inj:                                                         # модельный приток, поминутно: рост, плато, спад
            a, b = self.minute_index(_hm(day, it["start"])), self.minute_index(_hm(day, it["end"]))
            ramp, dec = int(it.get("ramp", 0)), int(it.get("decay", 0)); cols = [E.ST.index(s) for s in it["stations"] if s in E.ST]
            t = np.arange(max(a, 0), min(b + dec, len(M)))
            f = np.where(t < b, np.minimum(1.0, (t - a + 1) / ramp) if ramp else 1.0, np.maximum(0.0, 1 - (t - b + 1) / (dec + 1)))
            M[np.ix_(t, cols)] *= (1 + float(it["pct"]) / 100 * f)[:, None]
        self.M = M
        self.cs = np.vstack([np.zeros((1, E.NS)), np.cumsum(np.nan_to_num(M), 0)])
        self.nc = np.vstack([np.zeros((1, E.NS)), np.cumsum(np.isnan(M), 0)])

    def minute_index(self, ts):
        return int(np.floor((pd.Timestamp(ts) - self.t0) / pd.Timedelta(minutes=1)))

    def counts(self, a, b):
        """вход по станциям за минуты [a, b)"""
        a = min(max(a, 0), len(self.M)); b = min(max(b, a), len(self.M))
        return self.cs[b] - self.cs[a]

    def x_roll(self, X_full, c, off):
        """копия X_full, где последние 720 слотов до c — скользящие 15-минутные суммы, кончающиеся в момент GRID[c] + 15 + off минут"""
        X = X_full.copy(); end = self.minute_index(E.GRID[c] + pd.Timedelta(minutes=15 + off))
        ks = np.arange(max(self.s0, c - 720), c + 1); b = end - 15 * (c - ks); a = b - 15; ok = (a >= 0) & (b <= len(self.M))
        ks, a, b = ks[ok], a[ok], b[ok]
        X[ks] = np.where(self.nc[b] - self.nc[a] > 0, np.nan, self.cs[b] - self.cs[a])
        return X
