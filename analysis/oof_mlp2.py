"""MLP-2: та же сеть шире и дольше, цель — отклонение от нормы в log-шкале (log1p(y) - log1p(норма)). Добавляет "N2 MLP-остаток" в ../oof_nontree.pkl."""
import os, sys, time, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "app")); UP = os.path.dirname(ROOT)
import numpy as np, torch, torch.nn as nn
import engine as E
torch.set_num_threads(10); t0 = time.time()
w = E.World(); S = np.arange(E.T); F = w.feat_rows(w.X, S); y = w.X.reshape(-1); NS = E.NS
month = np.repeat(w.month, NS); slot = F["slot"].values.astype(int); st = F["st"].values.astype(int); cls = F["cls"].values.astype(int)
nrm = F["norm4"].values; have = (slot >= 14) & (slot <= 85) & ~np.isnan(y) & np.isin(month, (2, 5, 7, 9)) & ~np.isnan(nrm)
num = ["lag2", "lag3", "lag4", "lag6", "lag8", "lag96", "lag672", "norm4", "mean_2_5"]
Z = F[num].values.astype(float); Z = np.where(np.isnan(Z), nrm[:, None], Z); Zl = np.log1p(np.clip(Z, 0, None))
rel = Zl - np.log1p(np.clip(nrm, 0, None))[:, None]                     # лаги относительно нормы
flags = F[["off", "pre", "off_next", "off_prev"]].values.astype(float)
Xn = np.hstack([Zl, rel, flags]).astype("float32"); mu, sd = Xn[have].mean(0), Xn[have].std(0) + 1e-6; Xn = (Xn - mu) / sd
ln = np.log1p(np.clip(nrm, 0, None)); tgt = np.log1p(np.nan_to_num(y)) - ln
class Net(nn.Module):
    def __init__(s, k):
        super().__init__(); s.es = nn.Embedding(NS, 12); s.esl = nn.Embedding(96, 16); s.ec = nn.Embedding(7, 4)
        s.f = nn.Sequential(nn.Linear(k + 32, 512), nn.GELU(), nn.Dropout(0.15), nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))
    def forward(s, x, a, b, c): return s.f(torch.cat([x, s.es(a), s.esl(b), s.ec(c)], 1)).squeeze(1)
T = lambda arr, mk, dt=torch.float32: torch.tensor(arr[mk], dtype=dt)
pred = np.full(len(y), np.nan); EP = 40
for m in (2, 5, 7, 9):
    torch.manual_seed(0); tr = have & (month != m); te = have & (month == m)
    xt, at, bt, ct, yt = T(Xn, tr), T(st, tr, torch.long), T(slot, tr, torch.long), T(cls, tr, torch.long), T(tgt, tr)
    net = Net(Xn.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); n = len(yt); bs = 1024
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EP * ((n + bs - 1) // bs))
    for ep in range(EP):
        perm = torch.randperm(n); net.train()
        for i in range(0, n, bs):
            ix = perm[i:i + bs]; loss = (net(xt[ix], at[ix], bt[ix], ct[ix]) - yt[ix]).abs().mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad(): pred[te] = np.expm1(net(T(Xn, te), T(st, te, torch.long), T(slot, te, torch.long), T(cls, te, torch.long)).numpy() + ln[te])
    print(f"MLP-2 месяц {m}: WAPE {np.abs(y[te]-pred[te]).sum()/y[te].sum()*100:.2f}%, {time.time()-t0:.0f} с", flush=True)
d = pickle.load(open(os.path.join(UP, "oof_nontree.pkl"), "rb")); d["oof"]["N2 MLP-остаток"] = np.clip(pred, 0, None); pickle.dump(d, open(os.path.join(UP, "oof_nontree.pkl"), "wb"))
print(f"MLP-2 все: {np.abs(y[have]-pred[have]).sum()/y[have].sum()*100:.2f}%")
