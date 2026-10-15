"""
Audit of the gentrification pipeline: looks for mistakes, leakage and over-claims.

A. Response validity  - is the median-housing-cost column consistent across ACS releases (it moves in 2015)?
                        does the income response point in the right direction? top-coding?
B. Mean reversion      - how much of the accuracy comes from one or two baseline-level features?
C. Sanity              - shuffled labels must give ~50%
D. Label sensitivity   - absolute thresholds and extreme-tercile labels
E. Window sensitivity  - other pivot years / horizons
F. Error analysis      - accuracy by tract size and by county (out-of-fold predictions)

Usage: python audit.py     (needs results/ from gentrification.py; ~10 min)   Output: results/audit.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

import gentrification as g

warnings.filterwarnings("ignore")
res = json.load(open("results/results.json"))
cv5 = StratifiedKFold(5, shuffle=True, random_state=g.SEED)
OUT = {}


def xgb(resp):
    return g.SPECS["XGBoost"]["make"](**res[resp]["params"]["XGBoost"])


def l1(resp):
    return g.SPECS["L1 Logistic"]["make"](**res[resp]["params"]["L1 Logistic"])


def cv_acc(model, M, y):
    return float(cross_val_score(model, M, y, scoring="accuracy", cv=cv5, n_jobs=2).mean())


def say(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ data
say("loading ...")
acs = {y: g.load(y) for y in range(2010, 2017)}
d = g.build_dataset()
X, ids, idx = d["X"], d["ids"], d["X"].index
Y = {"cost": d["y_cost"], "income": d["y_inc"]}
POST = {"cost": d["cost_post"], "income": d["inc_post"]}

# ------------------------------------------------------------------ A. validity
say("\n[A1] median-housing-cost column: correlation of consecutive releases (a break would show as a dip at 2014->2015)")
A = {}
for y in range(2010, 2016):
    a = acs[y][g.cost_col(y)]
    b = acs[y + 1][g.cost_col(y + 1)]
    common = a.index.intersection(b.index)
    c = a[common].corr(b[common])
    A[f"{y}->{y + 1}"] = float(c)
    say(f"   {y}->{y + 1}: r = {c:.3f}   (columns {g.cost_col(y)} -> {g.cost_col(y + 1)})")
say("   share of tracts at the top-code (>= 4000):", {y: round(float((acs[y][g.cost_col(y)] >= 4000).mean()), 4) for y in (2012, 2016)})
OUT["A1_cost_column_consecutive_corr"] = A

mi = lambda y: acs[y]["S2503_C01_013E"]
dmed = (mi(2016) - mi(2012) * g.CPI[2016] / g.CPI[2012]).reindex(ids)
ok = dmed.notna()
r1, r2 = np.corrcoef(dmed[ok], POST["income"][ids][ok])[0, 1], spearmanr(dmed[ok], POST["income"][ids][ok])[0]
say(f"[A2] income response vs real change in median household income 2012->2016: pearson {r1:+.3f}, spearman {r2:+.3f} (should be clearly positive)")
OUT["A2_income_direction"] = dict(pearson=float(r1), spearman=float(r2))

# ------------------------------------------------------------------ B. baselines / mean reversion
say("\n[B] 5-fold CV accuracy of small baselines vs the full model")
F = pd.DataFrame({
    "med_cost_2011": acs[2011][g.cost_col(2011)],
    "med_inc_2011": acs[2011]["S2503_C01_013E"],
    "hellinger_2011": g.hellinger_to_affluent(acs[2011][g.INCOME_BINS]),
    "units_2011": acs[2011]["S2503_C01_001E"],
}).reindex(idx)
B = {}
for resp in ("cost", "income"):
    y = Y[resp][ids]
    sets = {"1 feature (2011 median cost)": ["med_cost_2011"], "1 feature (2011 income-distribution distance)": ["hellinger_2011"],
            "2 features (median cost + median income)": ["med_cost_2011", "med_inc_2011"]}
    row = {}
    for name, cols in sets.items():
        row[name] = cv_acc(g.pipe(LogisticRegression(max_iter=1000), scale=True), F.loc[ids, cols], y)
    imp = pd.read_csv(f"results/importance_{resp}.csv", index_col=0).iloc[:, 0].sort_values(ascending=False)
    top5 = [c for c in imp.index if c not in X.columns[-4:]][:5]
    row["top-5 XGBoost features"] = cv_acc(xgb(resp), X.loc[ids, top5], y)
    row["all 401 features, XGBoost"] = cv_acc(xgb(resp), X.loc[ids], y)
    row["all 401 features, L1 logit"] = cv_acc(l1(resp), X.loc[ids], y)
    B[resp] = row
    say(f"  {resp}: " + " | ".join(f"{k}: {v:.3f}" for k, v in row.items()))
r = np.corrcoef(F.loc[ids, "med_cost_2011"].fillna(F.med_cost_2011.median()), POST["cost"][ids])[0, 1]
say(f"  corr(2011 median cost, 2012->2016 real cost change) = {r:+.3f}   (negative = mean reversion)")
for f in ("units_2011", "med_cost_2011"):
    v = F.loc[ids, f].fillna(F[f].median())
    say(f"  single-feature AUC vs cost label, {f}: {max(roc_auc_score(Y['cost'][ids], v), 1 - roc_auc_score(Y['cost'][ids], v)):.3f}")
B["corr_level_vs_change_cost"] = float(r)
OUT["B_baselines"] = B

# ------------------------------------------------------------------ C. shuffled labels
say("\n[C] shuffled-label sanity check (should be about 0.50)")
C = {}
rng = np.random.default_rng(0)
for resp in ("cost", "income"):
    y = Y[resp][ids]
    accs = [cv_acc(xgb(resp), X.loc[ids], pd.Series(rng.permutation(y.values), index=ids)) for _ in range(3)]
    C[resp] = accs
    say(f"  {resp}: XGBoost on shuffled labels, 3 permutations: " + ", ".join(f"{a:.3f}" for a in accs))
OUT["C_shuffled"] = C

# ------------------------------------------------------------------ D. label sensitivity
say("\n[D] label sensitivity (5-fold CV; thresholds from the full valid sample, fine for a sensitivity check)")
D = {}
for resp in ("cost", "income"):
    post = POST[resp][ids]
    q1, q2 = post.quantile([1 / 3, 2 / 3])
    variants = {
        "baseline (median split)": (ids, Y[resp][ids]),
        "absolute (change > 0)": (ids, (post > 0).astype(int)),
        "extreme thirds (top vs bottom)": (post.index[(post <= q1) | (post >= q2)], (post[(post <= q1) | (post >= q2)] >= q2).astype(int)),
    }
    D[resp] = {}
    for name, (sub, y) in variants.items():
        nir = max(y.mean(), 1 - y.mean())
        D[resp][name] = dict(n=int(len(sub)), no_info=float(nir), xgb=cv_acc(xgb(resp), X.loc[sub], y), l1=cv_acc(l1(resp), X.loc[sub], y))
        say(f"  {resp:<7} {name:<32} n={len(sub)} NIR={nir:.3f}  XGB {D[resp][name]['xgb']:.3f}  L1 {D[resp][name]['l1']:.3f}")
OUT["D_label_sensitivity"] = D

# ------------------------------------------------------------------ E. window sensitivity
say("\n[E] other time windows (same hyper-parameters, 5-fold CV)")
E = {}
base = (g.PRE0, g.PRE1, g.PIVOT, g.END)
for name, win in {"features 2010-11, 2012->2015 (3-year horizon)": (2010, 2011, 2012, 2015),
                  "features 2011-12, 2013->2016 (pivot 2013)": (2011, 2012, 2013, 2016)}.items():
    g.PRE0, g.PRE1, g.PIVOT, g.END = win
    dd = g.build_dataset()
    E[name] = {}
    for resp, key in (("cost", "y_cost"), ("income", "y_inc")):
        y = dd[key][dd["ids"]]
        E[name][resp] = dict(no_info=float(max(y.mean(), 1 - y.mean())), xgb=cv_acc(xgb(resp), dd["X"].loc[dd["ids"]], y), l1=cv_acc(l1(resp), dd["X"].loc[dd["ids"]], y))
        say(f"  {name} | {resp:<7} XGB {E[name][resp]['xgb']:.3f}  L1 {E[name][resp]['l1']:.3f}")
g.PRE0, g.PRE1, g.PIVOT, g.END = base
OUT["E_windows"] = E

# ------------------------------------------------------------------ F. error analysis
say("\n[F] out-of-fold XGBoost accuracy by subgroup")
CNAME = {"001": "Alameda", "013": "Contra Costa", "019": "Fresno", "037": "Los Angeles", "059": "Orange", "065": "Riverside",
         "067": "Sacramento", "071": "San Bernardino", "073": "San Diego", "075": "San Francisco", "081": "San Mateo",
         "085": "Santa Clara", "111": "Ventura"}
Fr = {}
size = pd.qcut(F.loc[ids, "units_2011"], 3, labels=["small tracts", "medium tracts", "large tracts"])
county = pd.Series(ids.str[2:5], index=ids)
for resp in ("cost", "income"):
    y = Y[resp][ids]
    pred = pd.Series(cross_val_predict(xgb(resp), X.loc[ids], y, cv=cv5, n_jobs=2), index=ids)
    hit = (pred == y)
    by_size = hit.groupby(size).mean().round(3).to_dict()
    top = county.value_counts().head(10).index
    by_cty = {CNAME.get(c, c): round(float(hit[county == c].mean()), 3) for c in top}
    Fr[resp] = dict(overall=float(hit.mean()), by_size=by_size, by_county=by_cty, predicted_positive_rate=float(pred.mean()))
    say(f"  {resp}: overall {hit.mean():.3f}; by size {by_size}")
    say(f"     by county (10 largest): {by_cty}")
OUT["F_error_analysis"] = Fr

json.dump(OUT, open("results/audit.json", "w"), indent=1, default=float)
say("\nsaved results/audit.json")
