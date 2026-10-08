"""Обучение 4 моделей: для каждого месяца M модель учится на остальных трёх (только обычные рабочие дни, пассажирские слоты)
и используется для проигрывания дней месяца M. Запуск: python app/train_models.py (из корня репозитория)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, lightgbm as lgb
import engine as E

w = E.World(); S = np.arange(E.T)
F = w.feat_rows(w.X, S); y = w.X.reshape(-1); scope = np.repeat(w.scope_row, E.NS); month = np.repeat(w.month, E.NS)
P = dict(objective="regression_l1", learning_rate=0.08, num_leaves=63, min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
os.makedirs(E.MODELS, exist_ok=True); res = {}
for m in (2, 5, 7, 9):
    tr = scope & (month != m) & ~np.isnan(y); te = scope & (month == m) & ~np.isnan(y)
    mdl = lgb.train(P, lgb.Dataset(F[tr], y[tr], categorical_feature=["st"]), 300); mdl.save_model(os.path.join(E.MODELS, f"lgb_hold{m}.txt"))
    pr = mdl.predict(F[te]); res[m] = np.abs(y[te] - pr).sum() / y[te].sum() * 100; print(f"месяц {m}: обучено на {tr.sum():,} строках, WAPE в проверке (станции, рабочие дни) {res[m]:.2f}%", flush=True)
print("среднее WAPE: %.2f%%" % np.mean(list(res.values())))
