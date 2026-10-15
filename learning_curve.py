"""
Learning curve: how much training data do we need?

Trains XGBoost and L1-logistic regression (hyper-parameters taken from the full run in results/results.json)
on random subsets of the training tracts and scores them on the fixed validation + test tracts (never used for
tuning). Output: results/learning_curve.csv and results/fig_learning_curve.png

Usage:  python learning_curve.py          (needs results/results.json from gentrification.py; ~3 min)
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gentrification as g

FRACS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
REPS = 5
MODELS = ["XGBoost", "L1 Logistic"]

res = json.load(open("results/results.json"))
d = g.build_dataset()
X, tr, ev = d["X"][d["feats_all"]], d["tr_ids"], d["va_ids"].append(d["te_ids"])
rows = []
for resp, y in (("cost", d["y_cost"]), ("income", d["y_inc"])):
    nir = max(y[ev].mean(), 1 - y[ev].mean())
    for frac in FRACS:
        n = int(round(frac * len(tr)))
        for rep in range(1 if frac == 1.0 else REPS):
            sub = tr[np.random.default_rng(g.SEED + rep).permutation(len(tr))[:n]]
            for m in MODELS:
                model = g.SPECS[m]["make"](**res[resp]["params"][m]).fit(X.loc[sub], y[sub])
                acc = float((model.predict(X.loc[ev]) == y[ev].values).mean())
                rows.append(dict(response=resp, model=m, n_train=n, rep=rep, accuracy=acc, no_info_rate=nir))
        g.log(f"  {resp} n_train={n}: " + ", ".join(
            f"{m} {np.mean([r['accuracy'] for r in rows if r['response'] == resp and r['n_train'] == n and r['model'] == m]):.3f}"
            for m in MODELS))

df = pd.DataFrame(rows)
df.to_csv("results/learning_curve.csv", index=False)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, resp, title in zip(axs, ("cost", "income"), ("Response: change in housing cost", "Response: income-distribution shift")):
    for m, col in zip(MODELS, ("#2a6f97", "#e07a5f")):
        s = df[(df.response == resp) & (df.model == m)].groupby("n_train").accuracy.agg(["mean", "std"]).fillna(0)
        ax.plot(s.index, s["mean"], marker="o", color=col, label=m)
        ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=col, alpha=0.2)
    ax.axhline(df[df.response == resp].no_info_rate.iloc[0], color="#c1121f", ls="--", label="no-information rate")
    ax.set_xscale("log")
    ax.set_xlabel("training tracts (log scale)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
axs[0].set_ylabel(f"accuracy on {len(ev)} held-out tracts")
plt.tight_layout()
plt.savefig("results/fig_learning_curve.png", dpi=200)
print(df.groupby(["response", "model", "n_train"]).accuracy.mean().unstack("n_train").round(3).to_string())
