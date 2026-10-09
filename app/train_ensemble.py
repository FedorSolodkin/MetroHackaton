"""Обучение недостающих моделей ансамбля (LightGBM на остатке и MLP на отклонении от нормы), «месяц в проверку».
Модель M1 (LightGBM-L1) обучает app/train_models.py. Запуск из корня репозитория: python app/train_ensemble.py"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, lightgbm as lgb, torch
import engine as E
from ensemble import MLP, mlp_inputs

torch.set_num_threads(8); t0 = time.time()
w = E.World(); S = np.arange(E.T); F = w.feat_rows(w.X, S); y = w.X.reshape(-1); NS = E.NS
slot = np.repeat(w.slot, NS); month = np.repeat(w.month, NS); nrm = F["norm4"].values
have = (slot >= E.OPEN_SLOT) & (slot <= 85) & ~np.isnan(y) & np.isin(month, (2, 5, 7, 9)) & ~np.isnan(nrm)
os.makedirs(E.MODELS, exist_ok=True)
P = dict(objective="regression_l1", learning_rate=0.08, num_leaves=63, min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=6)
Xn = mlp_inputs(F, np.nan_to_num(nrm)); mu, sd = Xn[have].mean(0), Xn[have].std(0) + 1e-6; np.savez(os.path.join(E.MODELS, "mlp_scaler.npz"), mu=mu, sd=sd); Xn = (Xn - mu) / sd
ln = np.log1p(np.clip(np.nan_to_num(nrm), 0, None)); tgt = np.log1p(np.nan_to_num(y)) - ln
st_, sl_, cl_ = F["st"].values.astype(int), F["slot"].values.astype(int), F["cls"].values.astype(int)
T = lambda a, mk, dt=torch.float32: torch.tensor(a[mk], dtype=dt); EP, bs = 40, 1024
for m in (2, 5, 7, 9):
    tr = have & (month != m); te = have & (month == m)
    res = lgb.train(P, lgb.Dataset(F[tr], y[tr] - nrm[tr], categorical_feature=["st"]), 300); res.save_model(os.path.join(E.MODELS, f"lgb_res_hold{m}.txt"))
    pr = np.clip(nrm[te] + res.predict(F[te]), 0, None); w3 = np.abs(y[te] - pr).sum() / y[te].sum() * 100
    torch.manual_seed(0); xt, at, bt, ct, yt = T(Xn, tr), T(st_, tr, torch.long), T(sl_, tr, torch.long), T(cl_, tr, torch.long), T(tgt, tr)
    net = MLP(Xn.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); n = len(yt)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EP * ((n + bs - 1) // bs))
    for ep in range(EP):
        perm = torch.randperm(n); net.train()
        for i in range(0, n, bs):
            ix = perm[i:i + bs]; loss = (net(xt[ix], at[ix], bt[ix], ct[ix]) - yt[ix]).abs().mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval(); torch.save(net.state_dict(), os.path.join(E.MODELS, f"mlp_hold{m}.pt"))
    with torch.no_grad(): pn = np.expm1(net(T(Xn, te), T(st_, te, torch.long), T(sl_, te, torch.long), T(cl_, te, torch.long)).numpy() + ln[te])
    print(f"месяц {m}: LightGBM-остаток WAPE {w3:.2f}%, MLP-остаток WAPE {np.abs(y[te] - pn).sum() / y[te].sum() * 100:.2f}%, {time.time() - t0:.0f} с", flush=True)
