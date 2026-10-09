"""Chronos-2, дообученный (LoRA) на трёх месяцах и проверяемый на четвёртом, горизонт 30 мин, только история станции (контекст 672 слота).
Точки прогноза те же, что в oof_chronos2.py (каждый второй слот 06:30–00:15). Сохраняет ../oof_chronos2_ft.pkl после каждого месяца.
Запуск из корня репозитория (нужна видеокарта, пакет 64 укладывается в 4 ГБ)."""
import os, sys, time, pickle, shutil, gc
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "app")); UP = os.path.dirname(ROOT)
import numpy as np, torch
from chronos import Chronos2Pipeline
import engine as E

STEPS, LR, BS, CTX = 1000, 2e-5, 64, 672
w = E.World(); X = w.X; t0 = time.time()
tgt_all = np.where((w.slot >= 14) & (w.slot <= 85) & np.isin(w.month, (2, 5, 7, 9)) & ~np.isnan(X).any(axis=1) & (w.slot % 2 == 0))[0]; tgt_all = tgt_all[tgt_all - 2 - CTX >= 0]
base = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda"); qi = base.quantiles.index(0.5)
def ctx(c):
    out = []
    for j in range(E.NS):
        v = X[c - CTX + 1:c + 1, j].astype("float32"); bad = np.where(np.isnan(v))[0]; out.append(v[bad[-1] + 1:] if len(bad) else v)
    return out
keys_all, preds_all = [], []
for m in (2, 5, 7, 9):
    inputs = []
    for mm in (2, 5, 7, 9):
        if mm == m: continue
        idx = np.where((w.month == mm) & ~np.isnan(X).any(axis=1))[0]
        for j in range(E.NS): inputs.append(X[idx[0]:idx[-1] + 1, j].astype("float32"))
    ft = base.fit(inputs, prediction_length=2, finetune_mode="lora", context_length=CTX, learning_rate=LR, num_steps=STEPS, batch_size=BS,
                  output_dir=os.path.join(UP, "_ft_tmp"), remove_printer_callback=True)
    print(f"месяц {m}: дообучение готово, {time.time()-t0:.0f} с", flush=True)
    tg = tgt_all[w.month[tgt_all] == m]; items, keys = [], []
    for s in tg:
        c = s - 2
        for j, y in enumerate(ctx(c)):
            if len(y) >= 96: items.append(torch.tensor(y)); keys.append((s, j))
    pr = np.empty(len(items), dtype="float32"); B = 2048
    for i in range(0, len(items), B):
        res = ft.predict(items[i:i + B], prediction_length=2, batch_size=512); pr[i:i + B] = np.array([r[0, qi, 1].item() for r in res])
    keys_all += keys; preds_all.append(np.clip(pr, 0, None))
    yy = np.array([X[s, j] for s, j in keys]); print(f"месяц {m}: прогноз готов ({len(keys):,} точек), WAPE {np.abs(yy - pr).sum() / yy.sum() * 100:.2f}%, {time.time()-t0:.0f} с", flush=True)
    pickle.dump({"keys": np.array(keys_all), "chronos_ft": np.concatenate(preds_all)}, open(os.path.join(UP, "oof_chronos2_ft.pkl"), "wb"))
    del ft; gc.collect(); torch.cuda.empty_cache(); shutil.rmtree(os.path.join(UP, "_ft_tmp"), ignore_errors=True)
print("сохранено")
