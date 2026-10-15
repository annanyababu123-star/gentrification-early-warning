"""
Regenerate all figures from the saved outputs of gentrification.py (no model re-training).

Usage:  python make_figures.py [results_dir] [data_dir]
Needs:  <results>/results.json, importance_*.csv, labels.csv, <data>/tl_2016_06_tract.zip,
        <data>/acs_2012.csv, <data>/acs_2016.csv
Optional: feature_labels.csv (from fetch_labels.py) -> readable names on the importance chart.
"""
import json
import os
import sys
import textwrap

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

BLUE, RED, GREY = "#2a6f97", "#c1121f", "#6c757d"
INCOME_BINS = [f"S2503_C01_{i:03d}E" for i in range(2, 13)]
BIN_LABELS = ["<5k", "5-10k", "10-15k", "15-20k", "20-25k", "25-35k",
              "35-50k", "50-75k", "75-100k", "100-150k", "150k+"]
SPATIAL = ("spatial_lag_cost", "spatial_lag_income", "local_moran_cost", "local_moran_income")
RESP_TITLE = {"cost": "Response: change in housing cost", "income": "Response: income-distribution shift"}


def read_labels(out):
    df = pd.read_csv(f"{out}/labels.csv", dtype=str)
    return df.rename(columns={df.columns[0]: "GEOID"}).set_index("GEOID").astype(float)


def style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load_labels():
    for p in ("feature_labels.csv", "data/feature_labels.csv"):
        if os.path.exists(p):
            df = pd.read_csv(p).drop_duplicates("code").set_index("code")
            return df["label"].to_dict()
    return {}


SHORTEN = [
    ("MONTHLY HOUSING COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME IN THE PAST 12 MONTHS", "Housing cost share of income"),
    ("HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2011 INFLATION-ADJUSTED DOLLARS)", "Household income"),
    ("INCOME AND BENEFITS (IN 2011 INFLATION-ADJUSTED DOLLARS)", "Income"),
    ("MONTHLY HOUSING COSTS", "Monthly housing costs"),
    ("EDUCATIONAL ATTAINMENT OF HOUSEHOLDER", "Householder education"),
    ("YEAR HOUSEHOLDER MOVED INTO UNIT", "Year moved in"),
    ("EMPLOYMENT STATUS", "Employment"),
]


def nice(code, labels, width=60):
    """Readable axis label: tenure tag + the last path segments of the Census label, wrapped to two lines."""
    if code in SPATIAL:
        return code.replace("_", " ") + "  (engineered)"
    lab = labels.get(code, "")
    if not lab:
        return code
    for a, b in SHORTEN:
        lab = lab.replace(a, b)
    parts = [p for p in lab.split("!!") if p not in ("Estimate", "Percent")]
    tag = ""
    if parts and parts[0].lower().endswith("occupied housing units"):
        tag = {"owner": "Owner", "renter": "Renter", "occupied": "All"}.get(parts[0].split("-")[0].split()[0].lower(), "")
        parts = parts[1:]
    text = " > ".join(parts[-3:])
    return code + "\n" + textwrap.shorten((tag + ": " if tag else "") + text, width=width, placeholder="...").replace("$", r"\$")


def fig_accuracy(res, out):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, r in zip(axs, ("cost", "income")):
        t = pd.DataFrame(res[r]["test"])
        ax.bar(t["model"], t["accuracy"], color=BLUE,
               yerr=[t["accuracy"] - t["acc_ci_low"], t["acc_ci_high"] - t["accuracy"]], capsize=3,
               error_kw=dict(ecolor="#222", lw=1.2))
        ax.axhline(t["no_info_rate"].iloc[0], color=RED, ls="--", label="no-information rate")
        for i, a in enumerate(t["accuracy"]):
            ax.text(i, 0.415, f"{a:.1%}", ha="center", color="white", fontsize=9, fontweight="bold")
        ax.set_title(RESP_TITLE[r])
        ax.set_ylim(0.40, 0.85)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(loc="upper right", frameon=False)
    axs[0].set_ylabel(f"test accuracy (95% CI), n = {res['n_test']}")
    plt.tight_layout()
    plt.savefig(f"{out}/fig_accuracy.png", dpi=200)
    plt.close()


def fig_importance(out, labels):
    fig, axs = plt.subplots(2, 1, figsize=(9, 9))
    for ax, r in zip(axs, ("cost", "income")):
        imp = pd.read_csv(f"{out}/importance_{r}.csv", index_col=0).iloc[:, 0].sort_values(ascending=False)
        top = imp.head(10)[::-1]
        ax.barh([nice(c, labels) for c in top.index], top.values, color=BLUE)
        ax.set_title(f"Top 10 XGBoost features, {r} response (gain importance)", loc="left")
        ax.tick_params(axis="y", labelsize=8)
        ranks = {c: int(imp.index.get_loc(c)) + 1 for c in SPATIAL}
        names = [f"{k.replace('_', ' ')} #{v}" for k, v in ranks.items()]
        ax.set_xlabel("engineered spatial feature ranks (of %d):\n%s\n%s" % (len(imp), ", ".join(names[:2]), ", ".join(names[2:])),
                      fontsize=7.5, color=GREY)
    plt.tight_layout()
    plt.savefig(f"{out}/fig_importance.png", dpi=200)
    plt.close()


def fig_map(out, data):
    lab = read_labels(out)
    gdf = gpd.read_file(f"{data}/tl_2016_06_tract.zip").set_index("GEOID").join(lab, how="inner")
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, col, title in zip(axs, ("y_cost", "y_income"),
                              ("Housing cost rose faster than the median tract", "Income shifted toward affluence")):
        gdf.plot(color=np.where(gdf[col] == 1, RED, "#9fc5e8"), linewidth=0, ax=ax)
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
    fig.legend(handles=[Patch(color=RED, label="label 1: 'gentrifying' (2012-2016)"),
                        Patch(color="#9fc5e8", label="label 0")], loc="lower center", ncol=2, frameon=False)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(f"{out}/fig_map.png", dpi=200)
    plt.close()


def fig_income_example(out, data):
    lab = read_labels(out)
    cols = ["GEOID", "S2503_C01_001E"] + INCOME_BINS
    a12 = pd.read_csv(f"{data}/acs_2012.csv", usecols=cols, dtype={"GEOID": str}).set_index("GEOID")
    a16 = pd.read_csv(f"{data}/acs_2016.csv", usecols=cols, dtype={"GEOID": str}).set_index("GEOID")
    ok = lab.index[(a12.loc[lab.index, "S2503_C01_001E"] > 1000)]
    tract = lab.loc[ok, "income_shift"].sort_values(ascending=False).index[0]
    x = np.arange(11)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(x - 0.2, a12.loc[tract, INCOME_BINS].values, 0.4, label="2012", color=BLUE)
    ax.bar(x + 0.2, a16.loc[tract, INCOME_BINS].values, 0.4, label="2016", color="#e07a5f")
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=45)
    ax.set_ylabel("% of households")
    ax.set_title(f"Income distribution shift toward affluence, tract {tract}", fontsize=10)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{out}/fig_income_shift_example.png", dpi=200)
    plt.close()


def make_all(out="results", data="data"):
    style()
    res = json.load(open(f"{out}/results.json"))
    labels = load_labels()
    fig_accuracy(res, out)
    fig_importance(out, labels)
    fig_map(out, data)
    fig_income_example(out, data)


if __name__ == "__main__":
    make_all(*(sys.argv[1:3]))
