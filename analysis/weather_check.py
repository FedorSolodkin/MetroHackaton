import pandas as pd, numpy as np, lightgbm as lgb
a=pd.read_pickle("model_table.pkl")
cal_f=["st","slot","dow","off","pre","off_next","off_prev"]
lag=["lag2","lag3","lag4","lag6","lag8","lag96","lag672","mean_2_5","trend"]
rain=["precipitation","rain3","rain6","snowfall"]
allw=["temperature_2m","apparent_temperature","precipitation","rain3","rain6","snowfall","snow_depth","cloud_cover","wind_gusts_10m","relative_humidity_2m","dtemp3","weather_code"]
sets={"C0 календарь+лаги":cal_f+lag,"C1 +только осадки":cal_f+lag+rain,"C2 +вся погода":cal_f+lag+allw}
wape=lambda y,p:np.abs(y-p).sum()/y.sum()
P=dict(objective="regression_l1",learning_rate=0.08,num_leaves=63,min_data_in_leaf=60,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=4)
res={k:{} for k in sets}; parts=[]
for m in (2,5,7,9):
    tr,te=a[a.m!=m],a[a.m==m]
    for n,f in sets.items():
        mdl=lgb.train(P,lgb.Dataset(tr[f],tr.pax,categorical_feature=["st"]),300)
        p=mdl.predict(te[f]); res[n][m]=wape(te.pax.values,p)
        if n.startswith("C0"): parts.append(te[["ts","station","pax","rain3","snowfall"]].assign(p=p))
o=pd.DataFrame(res).T; o["среднее"]=o.mean(axis=1); print((o*100).round(1).to_string())
r=pd.concat(parts); r["rel"]=(r.pax-r.p)/r.p.clip(lower=20); r=r[r.p>150]
print("\nотн. остаток C0 по осадкам за 3ч (мм):")
print(r.groupby(pd.cut(r.rain3,[-1,0.05,0.5,2,5,100]),observed=True).rel.agg(["mean","median","count"]).round(3))
