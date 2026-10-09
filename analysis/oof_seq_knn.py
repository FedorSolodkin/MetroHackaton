"""Две новые разнородные модели, «месяц в проверку», станции × 15 мин, горизонт 30 мин (цель — слот s по данным до c=s-2):
Q — GRU по ВСЕЙ линии: последние L=32 слота (8 часов) по 19 станциям (отклонение от нормы и уровень в log-шкале) + календарь цели -> поправки для всех 19 станций сразу;
K — метод аналогов (kNN): ищем в обучающих месяцах похожие моменты (отклонения станции и хабов за последние 8 слотов, тот же режим дня и время) и берём то, что было дальше.
Сохраняет ../oof_new.pkl (плоские массивы T*19). Запуск из корня репозитория."""
import os, sys, time, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "app")); UP = os.path.dirname(ROOT)
import numpy as np, torch, torch.nn as nn
from sklearn.neighbors import KNeighborsRegressor
import engine as E

torch.set_num_threads(8); t0 = time.time()
w = E.World(); X = w.X; N = w.NORM; T, NS = E.T, E.NS; S = np.arange(T)
F = w.feat_rows(X, S); y = X.reshape(-1); month = np.repeat(w.month, NS); slotf = F["slot"].values.astype(int); nrm = F["norm4"].values
have = (slotf >= 14) & (slotf <= 85) & ~np.isnan(y) & np.isin(month, (2, 5, 7, 9)) & ~np.isnan(nrm)
Lx = np.log1p(np.clip(np.nan_to_num(X), 0, None)); Ln = np.log1p(np.clip(np.where(np.isnan(N), np.nan_to_num(X), N), 0, None)); R = Lx - Ln   # отклонение от нормы (log)
# ---------- Q: GRU ----------
L = 32; cls = w.cls; flags = np.column_stack([w.off, w.pre, w.off_next, w.off_prev]).astype("float32")
valid_s = np.array([(s - 2 - L + 1 >= 0) and not np.isnan(X[s - 2 - L + 1:s - 1]).any() and not np.isnan(X[s]).any() for s in S])
def window(s): c = s - 2; return np.concatenate([R[c - L + 1:c + 1], Lx[c - L + 1:c + 1] - Lx[c - L + 1:c + 1].mean(0)], axis=1)   # (L, 38)
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.gru = nn.GRU(2 * NS, 128, batch_first=True, num_layers=1); s.es = nn.Embedding(96, 16); s.ec = nn.Embedding(7, 4)
        s.head = nn.Sequential(nn.Linear(128 + 16 + 4 + 4 + NS, 256), nn.GELU(), nn.Dropout(0.15), nn.Linear(256, NS))
    def forward(s, xw, sl, cl, fl, ln): h = s.gru(xw)[1][-1]; return s.head(torch.cat([h, s.es(sl), s.ec(cl), fl, ln], 1))
pq = np.full((T, NS), np.nan); sl_all = w.slot.astype(int); ok_s = valid_s & (sl_all >= 14) & (sl_all <= 85) & np.isin(w.month, (2, 5, 7, 9)) & ~np.isnan(N).any(axis=1)
Wn = np.zeros((T, L, 2 * NS), dtype="float32"); 
for s in np.where(ok_s)[0]: Wn[s] = window(s)
mu, sd = Wn[ok_s].reshape(-1, 2 * NS).mean(0), Wn[ok_s].reshape(-1, 2 * NS).std(0) + 1e-6; Wn = (Wn - mu) / sd
tgt_rel = (Lx - Ln).astype("float32")                                             # цель: log1p(y) - log1p(норма)
for m in (2, 5, 7, 9):
    torch.manual_seed(0); tr = np.where(ok_s & (w.month != m))[0]; te = np.where(ok_s & (w.month == m))[0]
    tt = lambda ix: (torch.tensor(Wn[ix]), torch.tensor(sl_all[ix], dtype=torch.long), torch.tensor(cls[ix], dtype=torch.long), torch.tensor(flags[ix]), torch.tensor(Ln[ix].astype("float32") / 8))
    xtr = tt(tr); ytr = torch.tensor(tgt_rel[tr]); net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3); EP, bs = 60, 128
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EP * ((len(tr) + bs - 1) // bs))
    for ep in range(EP):
        net.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(tr), bs):
            ix = perm[i:i + bs]; loss = (net(*[a[ix] for a in xtr]) - ytr[ix]).abs().mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad(): rel = net(*tt(te)).numpy()
    pq[te] = np.expm1(rel + Ln[te])
    print(f"GRU месяц {m} готов, {time.time()-t0:.0f} с", flush=True)
# ---------- K: аналоги ----------
HUB = [0, 9, 18]; pk = np.full((T, NS), np.nan); KN = 30
def kfeat(s, j):
    c = s - 2; idx = np.arange(c - 7, c + 1)
    return np.concatenate([R[idx, j], R[idx[-3:]][:, HUB].reshape(-1), [np.sin(2 * np.pi * sl_all[s] / 96) * 2, np.cos(2 * np.pi * sl_all[s] / 96) * 2]])
for j in range(NS):
    rows = np.where(ok_s)[0]; Fm = np.array([kfeat(s, j) for s in rows]); yr = tgt_rel[rows, j]; wk = (cls[rows] >= 5); mo = w.month[rows]
    for m in (2, 5, 7, 9):
        for grp in (False, True):
            tr = (mo != m) & (wk == grp); te = (mo == m) & (wk == grp)
            if tr.sum() < KN or not te.any(): continue
            kn = KNeighborsRegressor(n_neighbors=KN, weights="distance").fit(Fm[tr], yr[tr]); pk[rows[te], j] = np.expm1(kn.predict(Fm[te]) + Ln[rows[te], j])
print(f"аналоги готовы, {time.time()-t0:.0f} с", flush=True)
fl = lambda a: np.clip(a.reshape(-1), 0, None)
oof = {"Q GRU по линии": fl(pq), "K аналоги": fl(pk)}
pickle.dump(oof, open(os.path.join(UP, "oof_new.pkl"), "wb"))
for k, v in oof.items(): mk = have & ~np.isnan(v); print(f"{k}: WAPE {np.abs(y[mk]-v[mk]).sum()/y[mk].sum()*100:.2f}% на {mk.sum():,} строках")
