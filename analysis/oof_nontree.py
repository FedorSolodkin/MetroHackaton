"""Модели «не деревья», месяц в проверку, станции × 15 мин, горизонт 30 мин (те же признаки, что у LightGBM приложения):
R  — Ridge отдельно для каждой станции (лаги, норма, one-hot слота и класса дня);
N  — нейросеть MLP (PyTorch, CPU): вложения станции/слота/класса дня + числовые признаки в log-шкале, цель — log1p(вход), потеря L1.
Сохраняет ../oof_nontree.pkl (плоские массивы T*19). Запуск из корня репозитория."""
import os, sys, time, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "app"))
import numpy as np, torch, torch.nn as nn
from sklearn.linear_model import Ridge
import engine as E

torch.set_num_threads(8); t0 = time.time()
w = E.World(); S = np.arange(E.T); F = w.feat_rows(w.X, S); y = w.X.reshape(-1); NS = E.NS
month = np.repeat(w.month, NS); slot = F["slot"].values.astype(int); st = F["st"].values.astype(int); cls = F["cls"].values.astype(int)
have = (slot >= 14) & (slot <= 85) & ~np.isnan(y) & np.isin(month, (2, 5, 7, 9)) & ~np.isnan(F["norm4"].values)
num = ["lag2", "lag3", "lag4", "lag6", "lag8", "lag96", "lag672", "norm4", "mean_2_5"]
Z = F[num].values.astype(float); nrm = F["norm4"].values
Z = np.where(np.isnan(Z), nrm[:, None], Z)                      # пропуски лагов (начало месяца) -> норма
Zl = np.log1p(np.clip(Z, 0, None)); flags = F[["off", "pre", "off_next", "off_prev"]].values.astype(float)
oof = {"R Ridge": np.full(len(y), np.nan), "N MLP": np.full(len(y), np.nan)}
# ---- Ridge по станциям ----
def design(mk):
    oh_slot = np.zeros((mk.sum(), 96)); oh_slot[np.arange(mk.sum()), slot[mk]] = 1
    oh_cls = np.zeros((mk.sum(), 7)); oh_cls[np.arange(mk.sum()), cls[mk]] = 1
    return np.hstack([Zl[mk], oh_slot, oh_cls, flags[mk]])
for m in (2, 5, 7, 9):
    for j in range(NS):
        tr = have & (month != m) & (st == j); te = have & (month == m) & (st == j)
        r = Ridge(alpha=1.0).fit(design(tr), np.log1p(y[tr])); oof["R Ridge"][te] = np.expm1(r.predict(design(te)))
print(f"Ridge готов, {time.time()-t0:.0f} с", flush=True)
# ---- MLP ----
class Net(nn.Module):
    def __init__(s, nnum):
        super().__init__(); s.es = nn.Embedding(NS, 8); s.esl = nn.Embedding(96, 12); s.ec = nn.Embedding(7, 4)
        s.f = nn.Sequential(nn.Linear(nnum + 24, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(s, xn, a, b, c): return s.f(torch.cat([xn, s.es(a), s.esl(b), s.ec(c)], 1)).squeeze(1)
Xn = np.hstack([Zl, flags]).astype("float32"); mu, sd = Xn[have].mean(0), Xn[have].std(0) + 1e-6; Xn = (Xn - mu) / sd
T = lambda a, mk, dt=torch.float32: torch.tensor(a[mk], dtype=dt)
for m in (2, 5, 7, 9):
    torch.manual_seed(0); tr = have & (month != m); te = have & (month == m)
    xt, at, bt, ct, yt = T(Xn, tr), T(st, tr, torch.long), T(slot, tr, torch.long), T(cls, tr, torch.long), torch.tensor(np.log1p(y[tr]), dtype=torch.float32)
    net = Net(Xn.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4); n = len(yt)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=12 * ((n + 2047) // 2048))
    for ep in range(12):
        perm = torch.randperm(n); net.train()
        for i in range(0, n, 2048):
            ix = perm[i:i + 2048]; loss = (net(xt[ix], at[ix], bt[ix], ct[ix]) - yt[ix]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad(): oof["N MLP"][te] = np.expm1(net(T(Xn, te), T(st, te, torch.long), T(slot, te, torch.long), T(cls, te, torch.long)).numpy())
    print(f"MLP месяц {m} готов, {time.time()-t0:.0f} с", flush=True)
for k in oof: oof[k] = np.clip(oof[k], 0, None)
pickle.dump({"oof": oof, "have": have}, open(os.path.join(os.path.dirname(ROOT), "oof_nontree.pkl"), "wb"))
wape = lambda k: np.abs(y[have] - oof[k][have]).sum() / y[have].sum() * 100
print({k: round(wape(k), 2) for k in oof}, f"всего {time.time()-t0:.0f} с")
