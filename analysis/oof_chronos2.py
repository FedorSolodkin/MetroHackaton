"""Chronos-2 (Amazon, Apache-2.0), без дообучения: прогноз входа станции на 2 слота вперёд (30 мин) по истории за 7 суток.
Вариант U: только история станции. Вариант C: + «норма» (медиана до 4 пред. дней того же класса) как прошлая и известная будущая ковариата.
Точки прогноза: каждый второй слот 06:30–00:15 всех дней 4 месяцев. Сохраняет ../oof_chronos2.pkl. Запуск из корня репозитория (нужна видеокарта)."""
import os, sys, time, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "app"))
import numpy as np, torch
from chronos import Chronos2Pipeline
import engine as E

w = E.World(); X, N = w.X, w.NORM; CTX = 672
tgt = np.where((w.slot >= 14) & (w.slot <= 85) & np.isin(w.month, (2, 5, 7, 9)) & ~np.isnan(X).any(axis=1) & (w.slot % 2 == 0))[0]
tgt = tgt[tgt - 2 - CTX >= 0]
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda"); qi = pipe.quantiles.index(0.5)
def ctx(arr, c):
    v = arr[c - CTX + 1:c + 1].astype("float32"); bad = np.where(np.isnan(v))[0]
    return v[bad[-1] + 1:] if len(bad) else v
items_u, items_c, keys = [], [], []
for s in tgt:
    c = s - 2; fut = np.array([w.norm(c + 1), w.norm(c + 2)])            # (2,19)
    for j in range(E.NS):
        y = ctx(X[:, j], c)
        if len(y) < 96: continue
        n = N[c - len(y) + 1:c + 1, j].astype("float32"); n = np.where(np.isnan(n), y, n)
        items_u.append(torch.tensor(y)); items_c.append({"target": y, "past_covariates": {"norm": n}, "future_covariates": {"norm": fut[:, j].astype("float32")}}); keys.append((s, j))
print("точек прогноза:", len(keys), flush=True)
out = {"keys": np.array(keys)}
for name, items in (("chronos_u", items_u), ("chronos_cov", items_c)):
    t0 = time.time(); preds = np.empty(len(items), dtype="float32"); B = 2048
    for i in range(0, len(items), B):
        res = pipe.predict(items[i:i + B], prediction_length=2, batch_size=512)
        preds[i:i + B] = np.array([r[0, qi, 1].item() for r in res])
        if i // B % 10 == 0: print(f"  {name}: {i + len(res):,}/{len(items):,}, {time.time() - t0:.0f} с", flush=True)
    out[name] = np.clip(preds, 0, None); print(f"{name} готов за {time.time() - t0:.0f} с", flush=True)
    pickle.dump(out, open(os.path.join(os.path.dirname(ROOT), "oof_chronos2.pkl"), "wb"))
print("сохранено")
