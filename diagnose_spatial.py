"""
Diagnostics: why did the engineered spatial features not help?

1. Do the 2010->2011 changes (the inputs of the spatial lag / Local Moran features) predict the 2012->2016 changes at all?
2. Are the 2012->2016 labels themselves spatially clustered? (global Moran's I)
3. Single-feature AUCs: spatial features vs. the strongest 2011 level features.
4. Alternative spatial context: neighbour averages of the top 2011 *level* features. Does that add accuracy?

Usage: python diagnose_spatial.py        (needs results/ from gentrification.py; ~5 min)
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score

import gentrification as g

res = json.load(open("results/results.json"))
d = g.build_dataset()
X, ids = d["X"], d["ids"]
idx = X.index
pos = idx.get_indexer(ids)
W, _ = g.queen_adjacency(idx)

a10, a11 = g.load(g.PRE0).loc[idx], g.load(g.PRE1).loc[idx]
cost_pre = a11[g.cost_col(g.PRE1)] - g.real_cost(a10, g.PRE0, g.PRE1)
inc_pre = g.hellinger_to_affluent(a10[g.INCOME_BINS]) - g.hellinger_to_affluent(a11[g.INCOME_BINS])
out = {}

# 1. does the pre-period change carry information about the post-period change?
print("\n[1] correlation of pre-period change (2010->2011) with post-period change (2012->2016)")
for name, pre, post in (("cost", cost_pre, d["cost_post"]), ("income", inc_pre, d["inc_post"])):
    ok = pre[ids].notna() & post[ids].notna()
    p, q = pre[ids][ok], post[ids][ok]
    print(f"  {name:<7} pearson {np.corrcoef(p, q)[0, 1]:+.3f}  spearman {spearmanr(p, q)[0]:+.3f}  (n={ok.sum()})")
    out[f"corr_pre_post_{name}"] = float(np.corrcoef(p, q)[0, 1])

# 2. are the labels spatially clustered? (global Moran's I, binary contiguity)
Wv = W[pos][:, pos]
print("\n[2] global Moran's I of the 2012->2016 labels (0 = random, >0 = clustered)")
for name, y in (("cost", d["y_cost"]), ("income", d["y_inc"])):
    z = y[ids].values - y[ids].mean()
    I = (len(z) / Wv.sum()) * (z @ (Wv @ z)) / (z @ z)
    print(f"  {name:<7} I = {I:+.3f}")
    out[f"moran_labels_{name}"] = float(I)

# 3. single-feature AUCs
print("\n[3] single-feature AUC (0.5 = useless); spatial features vs strongest level features")
for resp, y in (("cost", d["y_cost"]), ("income", d["y_inc"])):
    imp = pd.read_csv(f"results/importance_{resp}.csv", index_col=0).iloc[:, 0].sort_values(ascending=False)
    top = [c for c in imp.index if c not in X.columns[-4:]][:3]
    feats = list(X.columns[-4:]) + top
    row = {}
    for f in feats:
        v = X.loc[ids, f].fillna(X[f].median())
        a = roc_auc_score(y[ids], v)
        row[f] = max(a, 1 - a)
    row["own pre-change"] = (lambda v: max(roc_auc_score(y[ids], v), 1 - roc_auc_score(y[ids], v)))((cost_pre if resp == "cost" else inc_pre)[ids].fillna(0))
    print(f"  {resp}: " + "  ".join(f"{k}={v:.3f}" for k, v in row.items()))
    out[f"auc_{resp}"] = row

# 4. neighbour averages of the strongest 2011 level features
print("\n[4] 5-fold CV accuracy: baseline (401 features) vs + neighbour means of the top-10 level features")
county = pd.Series(ids.str[2:5], index=ids)
for resp, y in (("cost", d["y_cost"]), ("income", d["y_inc"])):
    imp = pd.read_csv(f"results/importance_{resp}.csv", index_col=0).iloc[:, 0].sort_values(ascending=False)
    top = [c for c in imp.index if c not in X.columns[-4:]][:10]
    lagged = pd.DataFrame({f"nbmean_{c}": g.nb_mean(W, X[c].values) for c in top}, index=idx)
    X2 = pd.concat([X, lagged], axis=1)
    model = g.SPECS["XGBoost"]["make"](**res[resp]["params"]["XGBoost"])
    row = {}
    for label, M in (("baseline", X), ("+ neighbour level means", X2)):
        r = cross_val_score(model, M.loc[ids], y[ids], scoring="accuracy", n_jobs=2, cv=StratifiedKFold(5, shuffle=True, random_state=g.SEED))
        c = cross_val_score(model, M.loc[ids], y[ids], scoring="accuracy", n_jobs=2, cv=GroupKFold(5), groups=county)
        row[label] = dict(random=float(r.mean()), county=float(c.mean()))
        print(f"  {resp:<7} {label:<26} random {r.mean():.3f}  county-blocked {c.mean():.3f}")
    out[f"neighbour_levels_{resp}"] = row

json.dump(out, open("results/diagnose_spatial.json", "w"), indent=1)
