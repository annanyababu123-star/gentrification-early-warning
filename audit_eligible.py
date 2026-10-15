"""
Follow-up audit: the classic gentrification population.

Gentrification is usually defined for tracts that START lower-income ("eligible" tracts). Our labels are relative changes over
all tracts, and mean reversion is strong (see audit.py). Here we check accuracy inside the eligible tracts, and for a
'classic' label = eligible AND cost rose faster than the median AND income shifted toward affluence faster than the median.

Usage: python audit_eligible.py    Output: results/audit_eligible.json
"""
import json
import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

import gentrification as g

warnings.filterwarnings("ignore")
res = json.load(open("results/results.json"))
cv5 = StratifiedKFold(5, shuffle=True, random_state=g.SEED)
d = g.build_dataset()
X, ids, idx = d["X"], d["ids"], d["X"].index
a11 = g.load(g.PRE1).loc[idx]
med_inc = a11["S2503_C01_013E"].reindex(ids)
med_cost = a11[g.cost_col(g.PRE1)].reindex(idx)
elig = ids[(med_inc < med_inc.median()).values]
out = {"n_eligible": int(len(elig)), "median_income_cutoff": float(med_inc.median())}
print(f"eligible tracts (2011 median household income below the median of ${med_inc.median():,.0f}): {len(elig)}", flush=True)

xgb = lambda r: g.SPECS["XGBoost"]["make"](**res[r]["params"]["XGBoost"])
l1 = lambda r: g.SPECS["L1 Logistic"]["make"](**res[r]["params"]["L1 Logistic"])
acc = lambda m, M, y: float(cross_val_score(m, M, y, scoring="accuracy", cv=cv5, n_jobs=2).mean())

for resp, y_all in (("cost", d["y_cost"]), ("income", d["y_inc"])):
    y = y_all[elig]
    nir = max(y.mean(), 1 - y.mean())
    one = acc(g.pipe(LogisticRegression(max_iter=1000), scale=True), med_cost.loc[elig].to_frame(), y)
    row = dict(positive_rate=float(y.mean()), no_info=float(nir), one_feature_median_cost=one,
               xgb=acc(xgb(resp), X.loc[elig], y), l1=acc(l1(resp), X.loc[elig], y))
    out[f"eligible_{resp}"] = row
    print(f"[eligible only] {resp:<7} positive rate {y.mean():.3f}  NIR {nir:.3f} | 1-feature {one:.3f} | XGB {row['xgb']:.3f} | L1 {row['l1']:.3f}", flush=True)

# classic label: eligible AND both responses above the median
yc = (d["y_cost"][ids] & d["y_inc"][ids]).astype(int) * pd.Series(ids.isin(elig).astype(int), index=ids)
nir = 1 - yc.mean()
print(f"[classic label] prevalence {yc.mean():.3f} (no-information accuracy {nir:.3f})", flush=True)
for name, model in (("XGBoost", xgb("cost")), ("L1 logit", l1("cost"))):
    p = cross_val_predict(model, X.loc[ids], yc, cv=cv5, method="predict_proba", n_jobs=2)[:, 1]
    hard = (p > 0.5).astype(int)
    # also a top-10% alert list: precision among the 10% of tracts with the highest predicted risk
    top = pd.Series(p, index=ids).sort_values(ascending=False).head(int(0.10 * len(ids))).index
    r = dict(auc=float(roc_auc_score(yc, p)), precision_at_0_5=float(precision_score(yc, hard, zero_division=0)),
             recall_at_0_5=float(recall_score(yc, hard, zero_division=0)), precision_top10pct=float(yc[top].mean()),
             base_rate=float(yc.mean()))
    pe, ye = pd.Series(p, index=ids)[elig], yc[elig]          # fair view: ineligible tracts are trivially label 0
    topE = pe.sort_values(ascending=False).head(int(0.10 * len(elig))).index
    r.update(auc_eligible_only=float(roc_auc_score(ye, pe)), base_rate_eligible=float(ye.mean()),
             precision_top10pct_eligible=float(ye[topE].mean()), lift_top10pct_eligible=float(ye[topE].mean() / ye.mean()))
    out[f"classic_{name}"] = r
    print(f"  {name:<8} ELIGIBLE ONLY: AUC {r['auc_eligible_only']:.3f} | top-10% list precision {r['precision_top10pct_eligible']:.3f} "
          f"vs base rate {r['base_rate_eligible']:.3f} (lift {r['lift_top10pct_eligible']:.2f}x)", flush=True)
    print(f"  {name:<8} AUC {r['auc']:.3f} | at 0.5: precision {r['precision_at_0_5']:.3f} recall {r['recall_at_0_5']:.3f} | "
          f"top-10% risk list: precision {r['precision_top10pct']:.3f} vs base rate {r['base_rate']:.3f}", flush=True)

json.dump(out, open("results/audit_eligible.json", "w"), indent=1)
