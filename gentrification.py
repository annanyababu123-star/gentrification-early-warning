#!/usr/bin/env python
"""
Early-warning system for gentrification in California census tracts.

Pipeline (see README.md):
  1. Load ACS 5-year tract tables (2010-2016) downloaded by fetch_census.py
  2. Build two gentrification responses between the pivot year (2012) and 2016
       - real change in median monthly housing cost
       - shift of the income distribution toward affluence (Hellinger distance)
  3. Build ~150-200 pre-period features (2011 ACS tables) + 4 engineered
     spatial features (spatial lag and Local Moran's I of the 2010->2011 change,
     queen contiguity from TIGER tract boundaries)
  4. Tune and compare Random Forest, XGBoost, L1-logistic regression and a
     majority-vote ensemble; evaluate against a no-information baseline
  5. Feature importance, spatial-feature ablation, county-blocked CV, figures

Run:  python gentrification.py            (outputs go to ./results)
"""
import json
import os
import re
import time
import warnings

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import (GroupKFold, ParameterGrid, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

FAST = os.environ.get("FAST", "0") == "1"   # quick demo mode: FAST=1 python gentrification.py
DATA, OUT = "data", ("results_fast" if FAST else "results")
SEED = 42
PRE0, PRE1 = 2010, 2011          # observation (feature) period
PIVOT, END = 2012, 2016          # prediction period: PIVOT -> END
# annual-average CPI-U (BLS) used to put costs in constant dollars
CPI = {2010: 218.056, 2011: 224.939, 2012: 229.594, 2013: 232.957,
       2014: 236.736, 2015: 237.017, 2016: 240.007}
INCOME_BINS = [f"S2503_C01_{i:03d}E" for i in range(2, 13)]   # 11 bins, % of households
BIN_LABELS = ["<5k", "5-10k", "10-15k", "15-20k", "20-25k", "25-35k",
              "35-50k", "50-75k", "75-100k", "100-150k", "150k+"]
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- data
def load(year):
    return pd.read_csv(f"{DATA}/acs_{year}.csv", dtype={"GEOID": str}).set_index("GEOID")


def cost_col(year):
    # the median monthly housing cost column moved in the 2015 release
    return "S2503_C01_028E" if year <= 2014 else "S2503_C01_024E"


def real_cost(df, year, to_year):
    return df[cost_col(year)] * CPI[to_year] / CPI[year]


def hellinger_to_affluent(bins):
    """Hellinger distance between a tract's income distribution and a baseline where
    everyone is in the top bin. Low = affluent, high = low-income."""
    tot = bins.sum(axis=1)
    p = bins.div(tot, axis=0).where(tot > 50)
    q = np.zeros(bins.shape[1])
    q[-1] = 1.0
    return pd.Series(np.sqrt(((np.sqrt(p.values) - np.sqrt(q)) ** 2).sum(axis=1) / 2), index=bins.index)


def queen_adjacency(index):
    gdf = gpd.read_file(f"{DATA}/tl_2016_06_tract.zip")
    gdf = gdf[gdf["GEOID"].isin(index)].set_index("GEOID").loc[index]
    g0 = gdf[["geometry"]].reset_index(drop=True)          # positional index == position in `index`
    j = gpd.sjoin(g0, g0, predicate="intersects")
    r = j.index.values
    c = j["index_right"].values
    m = r != c
    n = len(index)
    W = sp.csr_matrix((np.ones(m.sum()), (r[m], c[m])), shape=(n, n))
    W.data[:] = 1.0
    return W, gdf


def nb_mean(W, y):
    """Mean of y over each tract's neighbours (ignoring missing values)."""
    ok = ~np.isnan(y)
    num = W @ np.where(ok, y, 0.0)
    den = W @ ok.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def local_moran(W, y, train_mask):
    """Local Moran's I_i = z_i / m2 * (row-standardised neighbour mean of z); z is the deviation
    from the *training-sample* mean and m2 the training-sample mean of z^2."""
    mu = np.nanmean(y[train_mask])
    z = y - mu
    m2 = np.nanmean(z[train_mask] ** 2)
    return z / m2 * nb_mean(W, z)


# ------------------------------------------------------------------------ modelling
def pipe(est, scale=False):
    steps = [("imp", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("sc", StandardScaler()))
    return Pipeline(steps + [("m", est)])


SPECS = {
    "Random Forest": dict(
        make=lambda **p: pipe(RandomForestClassifier(random_state=SEED, n_jobs=1, **p)),
        grid=dict(n_estimators=[100, 300], max_features=[8, 16, 32, 64], min_samples_leaf=[1, 5, 15]),
        refine=["max_features", "min_samples_leaf"]),
    "XGBoost": dict(
        make=lambda **p: pipe(XGBClassifier(n_estimators=200, subsample=0.8, colsample_bytree=0.6,
                                            tree_method="hist", eval_metric="logloss",
                                            random_state=SEED, n_jobs=1, **p)),
        grid=dict(learning_rate=[0.05, 0.1, 0.25], max_depth=[2, 4, 6], reg_lambda=[1.0, 10.0, 50.0]),
        refine=["learning_rate", "max_depth", "reg_lambda"]),
    "L1 Logistic": dict(
        make=lambda **p: pipe(LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000,
                                                 random_state=SEED, **p), scale=True),
        grid=dict(C=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]),
        refine=["C"]),
}


if FAST:   # ~3-minute demo: tiny grids, no stage-2 refinement, no robustness section
    SPECS["Random Forest"]["grid"] = dict(n_estimators=[100], max_features=[16, 64], min_samples_leaf=[5])
    SPECS["XGBoost"]["grid"] = dict(learning_rate=[0.1], max_depth=[2, 4], reg_lambda=[10.0])
    SPECS["L1 Logistic"]["grid"] = dict(C=[0.03, 0.1, 0.3])
    for _s in SPECS.values():
        _s["refine"] = []


def search(make, grid, X, y, cv):
    best, rows = (-1, None), []
    for p in ParameterGrid(grid):
        s = cross_val_score(make(**p), X, y, cv=cv, scoring="accuracy", n_jobs=2).mean()
        rows.append({**p, "cv_acc": round(s, 4)})
        if s > best[0]:
            best = (s, p)
    return best, rows


def refine_grid(grid, best, names):
    """Second-stage grid: best value and the geometric midpoints toward its neighbours."""
    new = {k: [v] for k, v in best.items()}
    for k in names:
        vals = sorted(grid[k])
        v = best[k]
        i = vals.index(v)
        lo = vals[i - 1] if i > 0 else (v * v / vals[1] if isinstance(v, float) else v)
        hi = vals[i + 1] if i < len(vals) - 1 else (v * v / vals[-2] if isinstance(v, float) else v)
        cand = {v}
        for a in (lo, hi):
            g = np.sqrt(a * v)
            cand.add(int(round(g)) if isinstance(v, int) else float(f"{g:.3g}"))
        new[k] = sorted(cand)
    return new


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def metrics(name, y, pred, nir):
    n, k = len(y), int((y == pred).sum())
    lo, hi = wilson(k, n)
    return dict(model=name, n=n, accuracy=k / n, acc_ci_low=lo, acc_ci_high=hi,
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                pred_positive_rate=float(pred.mean()), no_info_rate=nir,
                p_vs_no_info=binomtest(k, n, nir, alternative="greater").pvalue,
                tn=int(((y == 0) & (pred == 0)).sum()), fp=int(((y == 0) & (pred == 1)).sum()),
                fn=int(((y == 1) & (pred == 0)).sum()), tp=int(((y == 1) & (pred == 1)).sum()))


def run_response(resp, X, y, tr, va, te, feats):
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    Xtr, ytr = X.loc[tr, feats], y.loc[tr]
    fitted, params, tuning = {}, {}, {}
    for name, sp_ in SPECS.items():
        t0 = time.time()
        (s1, p1), rows1 = search(sp_["make"], sp_["grid"], Xtr, ytr, cv)
        if sp_["refine"]:
            g2 = refine_grid(sp_["grid"], p1, sp_["refine"])
            (s2, p2), rows2 = search(sp_["make"], g2, Xtr, ytr, cv)
        else:
            (s2, p2), rows2 = (-1, p1), []
        p = p2 if s2 >= s1 else p1
        params[name] = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in p.items()}
        tuning[name] = dict(stage1=rows1, stage2=rows2, cv_acc=max(s1, s2))
        fitted[name] = sp_["make"](**p).fit(Xtr, ytr)
        log(f"  [{resp}] {name}: cv_acc={max(s1, s2):.3f} params={params[name]} ({time.time() - t0:.0f}s)")
    out = {}
    for split, idx in (("validation", va), ("test", te)):
        Xs, ys = X.loc[idx, feats], y.loc[idx].values
        nir = max(ys.mean(), 1 - ys.mean())
        preds = {n: m.predict(Xs) for n, m in fitted.items()}
        preds["Ensemble (vote)"] = (sum(preds.values()) >= 2).astype(int)
        out[split] = [metrics(n, ys, p, nir) for n, p in preds.items()]
    return fitted, params, tuning, out


# ---------------------------------------------------------------------------- main
def build_dataset():
    """Responses, features and train/validation/test split (shared by main() and learning_curve.py)."""
    log("Loading ACS tables ...")
    acs = {y: load(y) for y in (PRE0, PRE1, PIVOT, END)}
    idx = acs[PRE0].index
    for d in acs.values():
        idx = idx.intersection(d.index)
    idx = pd.Index(sorted(idx))
    acs = {y: d.loc[idx] for y, d in acs.items()}
    log(f"  tracts present in all four years: {len(idx)}")

    # ---- responses -------------------------------------------------------------
    cost_post = acs[END][cost_col(END)] - real_cost(acs[PIVOT], PIVOT, END)       # 2012 -> 2016 (2016 $)
    cost_pre = acs[PRE1][cost_col(PRE1)] - real_cost(acs[PRE0], PRE0, PRE1)        # 2010 -> 2011 (2011 $)
    H = {y: hellinger_to_affluent(acs[y][INCOME_BINS]) for y in (PRE0, PRE1, PIVOT, END)}
    inc_post = H[PIVOT] - H[END]        # > 0  => tract became more affluent
    inc_pre = H[PRE0] - H[PRE1]
    log(f"  corr(cost response, income response) = {cost_post.corr(inc_post):.3f}")

    # ---- spatial features --------------------------------------------------------
    log("Building queen-contiguity graph ...")
    W, gdf = queen_adjacency(idx)
    deg = np.asarray(W.sum(axis=1)).ravel()
    log(f"  edges={int(W.nnz // 2)}  mean degree={deg.mean():.2f}  isolated tracts={(deg == 0).sum()}")

    valid = cost_post.notna() & inc_post.notna()
    ids = idx[valid.values]
    tr_ids, tmp = train_test_split(ids, test_size=0.10, random_state=SEED)
    va_ids, te_ids = train_test_split(tmp, test_size=0.50, random_state=SEED)
    train_mask = idx.isin(tr_ids)

    thr_cost = cost_post[tr_ids].median()      # threshold learned on the training set only
    thr_inc = inc_post[tr_ids].median()
    y_cost = (cost_post > thr_cost).astype(int)
    y_inc = (inc_post > thr_inc).astype(int)
    log(f"  thresholds (train medians): cost {thr_cost:+.1f} $/month (real), income shift {thr_inc:+.4f}")

    spatial = pd.DataFrame({
        "spatial_lag_cost": nb_mean(W, cost_pre.values),
        "spatial_lag_income": nb_mean(W, inc_pre.values),
        "local_moran_cost": local_moran(W, cost_pre.values, train_mask),
        "local_moran_income": local_moran(W, inc_pre.values, train_mask),
    }, index=idx)

    # ---- tabular features (2011 ACS tables) -----------------------------------------------
    base = acs[PRE1]
    cols = [c for c in base.columns if re.match(r"(S2502|S2503|DP03|B25085)_", c)]
    base = base[cols]
    base = base.loc[:, (base.isna().mean() < 0.10) & (base.std() > 0)]
    corr = base.corr().abs()
    keep = []
    for c in base.columns:                                   # greedy de-duplication, |r| > 0.97
        if all(corr.loc[c, k] <= 0.97 for k in keep):
            keep.append(c)
    base = base[keep]
    X = pd.concat([base, spatial], axis=1)
    feats_all = list(X.columns)
    feats_nospatial = list(base.columns)
    log(f"  features: {len(feats_nospatial)} ACS + {spatial.shape[1]} engineered spatial = {len(feats_all)}")

    return dict(X=X, y_cost=y_cost, y_inc=y_inc, ids=ids, tr_ids=tr_ids, va_ids=va_ids, te_ids=te_ids,
                feats_all=feats_all, feats_nospatial=feats_nospatial, cost_post=cost_post, inc_post=inc_post,
                thr_cost=thr_cost, thr_inc=thr_inc)


def main():
    t_start = time.time()
    d = build_dataset()
    X, y_cost, y_inc, ids = d["X"], d["y_cost"], d["y_inc"], d["ids"]
    tr_ids, va_ids, te_ids = d["tr_ids"], d["va_ids"], d["te_ids"]
    feats_all, feats_nospatial = d["feats_all"], d["feats_nospatial"]
    cost_post, inc_post, thr_cost, thr_inc = d["cost_post"], d["inc_post"], d["thr_cost"], d["thr_inc"]

    results = dict(n_tracts=int(len(ids)), n_train=len(tr_ids), n_val=len(va_ids), n_test=len(te_ids),
                   n_features_acs=len(feats_nospatial), n_features_total=len(feats_all),
                   cost_threshold=float(thr_cost), income_threshold=float(thr_inc),
                   corr_responses=float(cost_post.corr(inc_post)),
                   positive_rate={"cost": float(y_cost[ids].mean()), "income": float(y_inc[ids].mean())})
    pd.DataFrame({"cost_change_real": cost_post, "income_shift": inc_post,
                  "y_cost": y_cost, "y_income": y_inc}).loc[ids].rename_axis("GEOID").to_csv(f"{OUT}/labels.csv")

    # ---- models ----------------------------------------------------------------------
    best_xgb, importances, fitted_all = {}, {}, {}
    for resp, y in (("cost", y_cost), ("income", y_inc)):
        log(f"Tuning and evaluating models: response = {resp}")
        fitted, params, tuning, ev = run_response(resp, X, y, tr_ids, va_ids, te_ids, feats_all)
        results[resp] = dict(params=params, tuning_cv=[{k: v["cv_acc"]} for k, v in tuning.items()],
                             validation=ev["validation"], test=ev["test"])
        json.dump(tuning, open(f"{OUT}/tuning_{resp}.json", "w"), indent=1, default=float)
        pd.DataFrame(ev["test"]).to_csv(f"{OUT}/metrics_test_{resp}.csv", index=False)
        pd.DataFrame(ev["validation"]).to_csv(f"{OUT}/metrics_validation_{resp}.csv", index=False)
        best_xgb[resp] = params["XGBoost"]
        fitted_all[resp] = fitted
        imp = pd.Series(fitted["XGBoost"].named_steps["m"].feature_importances_, index=feats_all)
        importances[resp] = imp.sort_values(ascending=False)
        importances[resp].to_csv(f"{OUT}/importance_{resp}.csv", header=["xgb_gain_importance"])
        for r in ev["test"]:
            log(f"    test {r['model']:<16} acc={r['accuracy']:.3f} prec={r['precision']:.3f} "
                f"rec={r['recall']:.3f} NIR={r['no_info_rate']:.3f} p={r['p_vs_no_info']:.3f}")

    # ---- robustness: spatial-feature ablation, random vs county-blocked CV ----------------------------
    log("Robustness checks (5-fold CV on all valid tracts) ...")
    county = pd.Series(ids.str[2:5], index=ids)
    rob = {}
    for resp, y in (("cost", y_cost), ("income", y_inc)):
        rob[resp] = {}
        for label, fs in ([] if FAST else (("with spatial features", feats_all), ("without spatial features", feats_nospatial))):
            model = SPECS["XGBoost"]["make"](**best_xgb[resp])
            a = cross_val_score(model, X.loc[ids, fs], y[ids], scoring="accuracy", n_jobs=2,
                                cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
            g = cross_val_score(model, X.loc[ids, fs], y[ids], scoring="accuracy", n_jobs=2,
                                cv=GroupKFold(5), groups=county)
            rob[resp][label] = dict(random_cv_mean=float(a.mean()), random_cv_std=float(a.std()),
                                    county_cv_mean=float(g.mean()), county_cv_std=float(g.std()))
            log(f"  {resp:<6} {label:<26} random-CV {a.mean():.3f}+-{a.std():.3f} | county-CV {g.mean():.3f}+-{g.std():.3f}")
        rob[resp]["no_info_rate_all"] = float(max(y[ids].mean(), 1 - y[ids].mean()))
    results["robustness"] = rob
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=1, default=float)

    # ---- figures (see make_figures.py) ---------------------------------------------------------
    log("Figures ...")
    import make_figures
    make_figures.make_all(OUT, DATA)
    log(f"Done in {(time.time() - t_start) / 60:.1f} min. Outputs in ./{OUT}")


if __name__ == "__main__":
    main()
