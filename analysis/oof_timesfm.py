"""TimesFM 2.5 (Google, 200M), без дообучения: прогноз входа станции на 2 слота вперёд по истории за 7 суток (как oof_chronos2.py, без ковариат).
Точки те же: каждый второй слот 06:30–00:15 4 месяцев. Сохраняет ../oof_timesfm.pkl. Запуск из корня репозитория."""
import os, sys, time, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "app")); UP = os.path.dirname(ROOT)
import numpy as np, torch, timesfm
import engine as E
w = E.World(); X = w.X; CTX = 672; t0 = time.time()
tgt = np.where((w.slot >= 14) & (w.slot <= 85) & np.isin(w.month, (2, 5, 7, 9)) & ~np.isnan(X).any(axis=1) & (w.slot % 2 == 0))[0]; tgt = tgt[tgt - 2 - CTX >= 0]
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(timesfm.ForecastConfig(max_context=CTX, max_horizon=2, normalize_inputs=True, use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
print(f"модель загружена, {time.time()-t0:.0f} с", flush=True)
items, keys = [], []
for s in tgt:
    c = s - 2
    for j in range(E.NS):
        v = X[c - CTX + 1:c + 1, j].astype("float32"); bad = np.where(np.isnan(v))[0]; v = v[bad[-1] + 1:] if len(bad) else v
        if len(v) >= 96: items.append(v); keys.append((s, j))
print("точек:", len(items), flush=True)
preds = np.empty(len(items), dtype="float32"); B = 1024
for i in range(0, len(items), B):
    pf, qf = model.forecast(horizon=2, inputs=items[i:i + B]); preds[i:i + B] = pf[:, 1]
    if (i // B) % 10 == 0: print(f"  {i + len(pf):,}/{len(items):,}, {time.time()-t0:.0f} с", flush=True)
pickle.dump({"keys": np.array(keys), "timesfm": np.clip(preds, 0, None)}, open(os.path.join(UP, "oof_timesfm.pkl"), "wb"))
yy = np.array([X[s, j] for s, j in keys]); print(f"сохранено; WAPE {np.abs(yy - preds).sum() / yy.sum() * 100:.2f}%, всего {time.time()-t0:.0f} с")
