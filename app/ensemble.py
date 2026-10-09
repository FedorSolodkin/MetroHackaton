"""Ансамбль прогнозов входа на станции (простое среднее): проверен «месяц в проверку» (WAPE 7.95% против 9.74% у нормы на тех же строках).

Состав по умолчанию (5 моделей, выбраны за разнообразие ошибок):
  M1  LightGBM (L1) на признаках лагов/нормы/календаря;
  M3  LightGBM на остатке от нормы (y - норма);
  N2  MLP на отклонении от нормы (log-шкала), вложения станции/слота/класса дня;
  M5  инерция: норма × (1 + 0.7 × отклонение последних 30 минут);
  CC  Chronos-2 (Amazon, без дообучения) с нормой как ковариатой — если есть видеокарта и пакет chronos (METRO_CHRONOS=0 отключает).
Для каждого проверочного месяца используются модели, обученные на остальных трёх месяцах (honest out-of-sample).
"""
import os

import numpy as np
import torch
import torch.nn as nn

import engine as E

HERE = os.path.dirname(os.path.abspath(__file__)); MODELS = os.path.join(HERE, "models")
NUM = ["lag2", "lag3", "lag4", "lag6", "lag8", "lag96", "lag672", "norm4", "mean_2_5"]
FLAGS = ["off", "pre", "off_next", "off_prev"]


class MLP(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.es = nn.Embedding(E.NS, 12); self.esl = nn.Embedding(96, 16); self.ec = nn.Embedding(7, 4)
        self.f = nn.Sequential(nn.Linear(k + 32, 512), nn.GELU(), nn.Dropout(0.15), nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, x, a, b, c):
        return self.f(torch.cat([x, self.es(a), self.esl(b), self.ec(c)], 1)).squeeze(1)


def mlp_inputs(F, nrm):
    """Признаки для MLP из таблицы признаков F и нормы nrm (без NaN)."""
    Z = F[NUM].values.astype(float); Z = np.where(np.isnan(Z), nrm[:, None], Z); Zl = np.log1p(np.clip(Z, 0, None))
    rel = Zl - np.log1p(np.clip(nrm, 0, None))[:, None]
    return np.hstack([Zl, rel, F[FLAGS].values.astype(float)]).astype("float32")


class Ensemble:
    def __init__(self, world, use_chronos="auto"):
        import lightgbm as lgb
        self.w = world; self.months = sorted(world.models)
        self.res = {m: lgb.Booster(model_file=os.path.join(MODELS, f"lgb_res_hold{m}.txt")) for m in self.months}
        sc = np.load(os.path.join(MODELS, "mlp_scaler.npz")); self.mu, self.sd = sc["mu"], sc["sd"]
        self.mlp = {}
        for m in self.months:
            net = MLP(len(self.mu)); net.load_state_dict(torch.load(os.path.join(MODELS, f"mlp_hold{m}.pt"), map_location="cpu")); net.eval(); self.mlp[m] = net
        self.chronos = None; self.chronos_note = "выключен"
        env = os.environ.get("METRO_CHRONOS", "auto").lower()
        if env != "0" and use_chronos != "0":
            try:
                if torch.cuda.is_available() or env == "1":
                    from chronos import Chronos2Pipeline
                    self.chronos = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda" if torch.cuda.is_available() else "cpu")
                    self.q50 = self.chronos.quantiles.index(0.5); self.chronos_note = "Chronos-2 (" + ("GPU" if torch.cuda.is_available() else "CPU") + ")"
                else:
                    self.chronos_note = "нет видеокарты"
            except Exception as e:  # noqa: BLE001
                self.chronos_note = f"недоступен ({type(e).__name__})"
        self.names = ["LightGBM", "LightGBM-остаток", "MLP-остаток", "Инерция"] + (["Chronos-2"] if self.chronos is not None else [])

    # ---------- компоненты на один целевой слот s (19 станций) ----------
    def components(self, month, F, nrm, V, s):
        w = self.w; out = {}
        out["LightGBM"] = np.clip(w.models[month].predict(F), 0, None)
        out["LightGBM-остаток"] = np.clip(nrm + self.res[month].predict(F), 0, None)
        xn = (mlp_inputs(F, nrm) - self.mu) / self.sd; ln = np.log1p(np.clip(nrm, 0, None))
        with torch.no_grad():
            r = self.mlp[month](torch.tensor(xn), torch.tensor(F["st"].values, dtype=torch.long), torch.tensor(F["slot"].values, dtype=torch.long), torch.tensor(F["cls"].values, dtype=torch.long)).numpy()
        out["MLP-остаток"] = np.clip(np.expm1(r + ln), 0, None)
        a = np.nan_to_num(V[s - 2] + V[s - 3]); n = np.nan_to_num(w.NORM[s - 2] + w.NORM[s - 3]); n = np.where(n <= 0, nrm * 2, n)
        dev = np.clip(a / np.clip(n, 1, None) - 1, -0.6, 1.0); out["Инерция"] = np.clip(nrm * (1 + 0.7 * dev), 0, None)
        return out

    def chronos_forecast(self, X_live, c, horizon=E.HORIZON):
        """Chronos-2 с нормой как ковариатой: (horizon, 19) или None."""
        if self.chronos is None:
            return None
        w = self.w; CTX = 672; items = []
        fut = np.array([w.norm(c + k) for k in range(1, horizon + 1)])                       # (horizon,19)
        for j in range(E.NS):
            y = X_live[c - CTX + 1:c + 1, j].astype("float32"); bad = np.where(np.isnan(y))[0]; y = y[bad[-1] + 1:] if len(bad) else y
            n = w.NORM[c - len(y) + 1:c + 1, j].astype("float32"); n = np.where(np.isnan(n), y, n)
            items.append({"target": y, "past_covariates": {"norm": n}, "future_covariates": {"norm": np.nan_to_num(fut[:, j]).astype("float32")}})
        res = self.chronos.predict(items, prediction_length=horizon, batch_size=32)
        return np.clip(np.array([r[0, self.q50, :].cpu().numpy() for r in res]).T, 0, None)   # (horizon,19)

    def forecast(self, month, X_live, c, horizon=E.HORIZON):
        """Рекурсивный прогноз слотов c+1..c+horizon (среднее компонентов). Возвращает (horizon,19) и разброс моделей."""
        w = self.w; V = X_live.copy(); V[c + 1:c + horizon + 1] = np.nan; ch = self.chronos_forecast(X_live, c, horizon)
        saveX = w.X; w.X = X_live; out = np.zeros((horizon, E.NS)); spread = np.zeros((horizon, E.NS))
        try:
            for k in range(1, horizon + 1):
                s = c + k; F = w.feat_rows(V, np.array([s])); nrm = w.norm(s)
                comp = self.components(month, F, nrm, V, s)
                if ch is not None: comp["Chronos-2"] = ch[k - 1]
                A = np.stack(list(comp.values())); out[k - 1] = A.mean(0); spread[k - 1] = A.std(0); V[s] = out[k - 1]
        finally:
            w.X = saveX
        return out, spread
