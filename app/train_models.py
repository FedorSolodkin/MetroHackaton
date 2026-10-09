"""Обучение 4 моделей «месяц в проверку»: для месяца M модель учится на остальных трёх (ВСЕ дни: рабочие, выходные, праздничные;
пассажирские слоты 06:30–00:30) и используется для проигрывания дней месяца M. Запуск из корня репозитория: python app/train_models.py
Печатает ошибку (WAPE) отдельно: рабочие дни / суббота / воскресенье / праздничные и предпраздничные будни, и сравнение с нормой."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, lightgbm as lgb
import engine as E

w = E.World(); S = np.arange(E.T)
F = w.feat_rows(w.X, S); y = w.X.reshape(-1)
slot_ok = np.repeat((w.slot >= E.OPEN_SLOT) & (w.slot <= 85), E.NS); month = np.repeat(w.month, E.NS); cls = np.repeat(w.cls, E.NS); off = np.repeat(w.off, E.NS); pre = np.repeat(w.pre, E.NS)
dow = np.repeat(w.dow, E.NS); norm = F["norm4"].values
have = slot_ok & ~np.isnan(y) & np.isin(month, (2, 5, 7, 9))
grp = {"рабочие дни (пн–пт)": (dow < 5) & (off == 0) & (pre == 0), "суббота (обычная)": (dow == 5) & (off == 0), "воскресенье (обычное)": (dow == 6) & (off == 0),
       "праздничные и предпраздничные будни": (dow < 5) & ((off == 1) | (pre == 1)), "выходные в праздничные блоки (кроме обычных)": (dow >= 5) & (off == 1) & False}
# выходные дни праздничных блоков: 21–22 фев, 2–3 и 9–10 мая
blk = np.isin(w.svc.strftime("%m-%d"), ["02-21", "02-22", "05-02", "05-03", "05-09", "05-10"]); blkr = np.repeat(blk, E.NS)
grp["суббота/воскресенье в праздничных блоках"] = blkr & (dow >= 5); grp["суббота (обычная)"] = (dow == 5) & ~blkr; grp["воскресенье (обычное)"] = (dow == 6) & ~blkr; grp.pop("выходные в праздничные блоки (кроме обычных)")
P = dict(objective="regression_l1", learning_rate=0.08, num_leaves=63, min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
os.makedirs(E.MODELS, exist_ok=True); pred = np.full(len(y), np.nan)
for m in (2, 5, 7, 9):
    tr = have & (month != m); te = have & (month == m)
    mdl = lgb.train(P, lgb.Dataset(F[tr], y[tr], categorical_feature=["st"]), 300); mdl.save_model(os.path.join(E.MODELS, f"lgb_hold{m}.txt")); pred[te] = mdl.predict(F[te])
    print(f"месяц {m}: обучено на {tr.sum():,} строках, проверка {te.sum():,}", flush=True)
wape = lambda a, b: np.abs(a - b).sum() / a.sum() * 100
print("\nWAPE, % (станции × 15 мин, месяц в проверку)             модель | норма (до 4 пред. дней того же класса)   | строк")
for name, mk in {**grp, "ВСЕ дни": np.ones(len(y), bool)}.items():
    k = have & mk & ~np.isnan(norm); print(f"  {name:46s} {wape(y[k], pred[k]):6.2f} | {wape(y[k], norm[k]):6.2f}   | {k.sum():,}")
print("\nпо месяцам, рабочие дни:", {m: round(wape(y[have & grp['рабочие дни (пн–пт)'] & (month == m)], pred[have & grp['рабочие дни (пн–пт)'] & (month == m)]), 2) for m in (2, 5, 7, 9)})
print("по месяцам, все дни:   ", {m: round(wape(y[have & (month == m)], pred[have & (month == m)]), 2) for m in (2, 5, 7, 9)})
