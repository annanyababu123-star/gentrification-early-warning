"""Builds writeup.pdf (2 pages) from results/results.json. Re-run after fetch_labels.py to get readable feature names."""
import json
import sys

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import make_figures as mf

OUT = "results"
res = json.load(open(f"{OUT}/results.json"))
labels = mf.load_labels()
TEAM = sys.argv[1] if len(sys.argv) > 1 else "Team 40: Annanya Babu (PES1UG24AM045), Aayushman Singh (PES1UG24AM006)"

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.2, leading=10.0, spaceAfter=2)
h = ParagraphStyle("h", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10, spaceBefore=5, spaceAfter=2,
                   textColor=colors.HexColor("#1d3557"))
title = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=14, spaceAfter=1)
sub = ParagraphStyle("s", parent=body, alignment=1, fontSize=8.5, textColor=colors.HexColor("#444444"))
bul = ParagraphStyle("bu", parent=body, leftIndent=10, bulletIndent=0)
cap = ParagraphStyle("c", parent=body, fontSize=7.6, leading=9, textColor=colors.HexColor("#444444"))


def P(t, st=body):
    return Paragraph(t, st)


def B(t):
    return Paragraph(t, bul, bulletText="•")


def tm(resp, model):
    return next(r for r in res[resp]["test"] if r["model"] == model)


def pct(x):
    return f"{100 * x:.1f}%"


def top_feats(resp, k=3):
    imp = pd.read_csv(f"{OUT}/importance_{resp}.csv", index_col=0).iloc[:, 0].sort_values(ascending=False)
    return [mf.nice(c, labels, width=70) for c in imp.index[:k]], {c: int(imp.index.get_loc(c)) + 1 for c in mf.SPATIAL}, len(imp)


cf, crank, nfeat = top_feats("cost")
inf, irank, _ = top_feats("income")
rob = res["robustness"]
au = json.load(open(f"{OUT}/audit.json"))
ae = json.load(open(f"{OUT}/audit_eligible.json"))
dl = au["D_label_sensitivity"]
dg = json.load(open(f"{OUT}/diagnose_spatial.json"))
_d = [v["+ neighbour level means"][k] - v["baseline"][k] for v in (dg["neighbour_levels_cost"], dg["neighbour_levels_income"]) for k in ("random", "county")]
_sp = max(dg["auc_cost"][k] for k in mf.SPATIAL)
why = (f"Diagnostics (diagnose_spatial.py): the 2010\u21922011 changes that feed the spatial features barely predict the 2012\u21922016 change "
       f"(correlation {dg['corr_pre_post_cost']:+.2f} for cost, {dg['corr_pre_post_income']:+.2f} for income, as expected when ACS windows overlap), and the 2011 levels already carry the signal "
       f"(single-feature AUC {dg['auc_cost']['S2503_C01_028E']:.2f} for median housing cost vs at most {_sp:.2f} for any spatial feature). "
       f"The labels do cluster in space (Moran's I {dg['moran_labels_cost']:.2f} cost, {dg['moran_labels_income']:.2f} income), but adding neighbour averages of the top-10 level features "
       f"moves CV accuracy by only {100 * min(_d):+.1f} to {100 * max(_d):+.1f} points, within noise.")
lc = pd.read_csv(f"{OUT}/learning_curve.csv")
lcm = lc.groupby(["response", "n_train"]).accuracy.mean()


def plateau(resp):
    m = lcm[resp]
    n = int(m[m >= m.max() - 0.01].index[0])
    return n, float(m.loc[n]), float(m.iloc[-1])


n_c, a_c, f_c = plateau("cost")
n_i, a_i, f_i = plateau("income")
R = res
nir_c, nir_i = tm("cost", "XGBoost")["no_info_rate"], tm("income", "XGBoost")["no_info_rate"]

story = [
    P("Predicting Gentrification in California Census Tracts: An Early-Warning System", title),
    P(TEAM + "  |  UE24CS352A Machine Learning mini-project", sub),
    P("1. Problem statement", h),
    P("Rising housing costs push low-income households out of California's neighbourhoods. Policymakers need to know "
      "<i>before</i> it happens which census tracts are likely to gentrify. We cast this as binary classification: "
      "given a tract's characteristics in 2010-2011, predict whether it gentrifies between 2012 and 2016. "
      "Gentrification is operationalised in two ways: (i) <b>cost</b>: the tract's real median monthly housing cost rises "
      "more than that of the median tract, and (ii) <b>income</b>: its income distribution shifts toward affluence more than "
      "the median tract's. The approach follows the early-warning-system framing of Pattabi (Stanford, 2018)."),
    P("2. Dataset", h),
    P(f"<b>American Community Survey (ACS) 5-year estimates</b> at census-tract level for California, releases 2010-2016, pulled from the "
      f"Census Bureau API (the successor of American FactFinder): tables S2502, S2503 (occupied housing units by demographics and "
      f"income/housing cost), DP03 (economic profile) and B25085. <b>TIGER/Line 2016</b> tract boundaries provide neighbours. "
      f"{R['n_tracts']:,} tracts have both responses defined (8,056 appear in all needed years; the rest lack a usable cost or income estimate). "
      f"Split: {R['n_train']:,} train / {R['n_val']} validation / {R['n_test']} test (random, seed 42). "
      f"Each tract is described by {R['n_features_acs']} ACS variables from 2011 plus 4 engineered spatial features."),
    P("3. Approach", h),
    B("<b>Responses (2012 to 2016).</b> Cost: 2016 median monthly housing cost minus the 2012 value converted to 2016 dollars with CPI-U. "
      "Income: for each tract we take the 11-bin household-income distribution P and compute the Hellinger distance "
      "H(P, Q) = sqrt( 0.5 · sum<sub>i</sub> ( sqrt(P<sub>i</sub>) − sqrt(Q<sub>i</sub>) )<super>2</super> ) to a baseline Q in which every household is in the top bin; the response is H(2012) − H(2016), "
      "so positive means more affluent. Both are labelled 1 if above the <i>training-set median</i>, giving balanced classes."),
    B("<b>Features.</b> 2011 ACS tables (columns with ≥10% missing, constants and near-duplicates with |r|>0.97 removed; median imputation). "
      "Following the spatial-equilibrium theory of endogenous gentrification, we add a <b>spatial lag</b> and <b>Local Moran's I</b> of the "
      "2010→2011 change in cost and in income distribution, computed on a queen-contiguity tract graph (25,763 edges, mean degree 6.4)."),
    B("<b>Models.</b> Random Forest, XGBoost, L1-penalised logistic regression (LASSO) and a majority-vote ensemble of the three. "
      "Hyper-parameters (forest size/features per split/leaf size; learning rate/depth/λ; C) are tuned by a two-stage grid search "
      "(coarse grid, then a refined grid around the best point) using 3-fold CV accuracy on the training set only. "
      "The test set is used once, for the final numbers."),
    B("<b>Evaluation.</b> Accuracy with 95% Wilson interval, precision and recall, compared with the no-information rate (share of the majority class) "
      "via an exact binomial test. Two robustness checks on all tracts: removing the spatial features (ablation) and 5-fold CV "
      "blocked by county, which stops neighbouring tracts leaking across folds."),
    P("4. Implementation overview", h),
    P("Python (pandas, scikit-learn, XGBoost, GeoPandas). <font face='Courier'>fetch_census.py</font> downloads the data; "
      "<font face='Courier'>gentrification.py</font> builds responses and features, tunes and evaluates the models and writes metrics and figures to "
      "<font face='Courier'>results/</font>; <font face='Courier'>make_figures.py</font> redraws the figures. A <font face='Courier'>FAST=1</font> mode runs the whole "
      "pipeline in about 3 minutes (full run: about 45 minutes on 2 cores). Everything is seeded and documented in the README."),
    P("5. Results", h),
]

hdr = ["Model", "Cost: acc. [95% CI]", "Prec.", "Rec.", "Income: acc. [95% CI]", "Prec.", "Rec."]
rows = [hdr]
for m in ("Random Forest", "XGBoost", "L1 Logistic", "Ensemble (vote)"):
    c, i = tm("cost", m), tm("income", m)
    rows.append([m, f"{pct(c['accuracy'])} [{pct(c['acc_ci_low'])}, {pct(c['acc_ci_high'])}]", f"{c['precision']:.2f}", f"{c['recall']:.2f}",
                 f"{pct(i['accuracy'])} [{pct(i['acc_ci_low'])}, {pct(i['acc_ci_high'])}]", f"{i['precision']:.2f}", f"{i['recall']:.2f}"])
rows.append(["No-information rate", pct(nir_c), "", "", pct(nir_i), "", ""])
t = Table(rows, colWidths=[3.3 * cm, 3.9 * cm, 1.2 * cm, 1.2 * cm, 3.9 * cm, 1.2 * cm, 1.2 * cm])
t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.8), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                       ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f0")), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                       ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Oblique"),
                       ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
story += [t, P(f"Test set, n = {R['n_test']}. All models beat the no-information rate (p &lt; 0.001), but the intervals overlap, so the models are not statistically distinguishable.", cap),
          Spacer(1, 3), Image(f"{OUT}/fig_accuracy.png", width=10.8 * cm, height=4.3 * cm), Spacer(1, 2),
          Image(f"{OUT}/fig_learning_curve.png", width=10.8 * cm, height=4.1 * cm),
          P("Learning curve: accuracy on 797 held-out tracts when training on random subsets of the training tracts (mean \u00b1 sd over 5 draws).", cap)]

wc, woc = rob["cost"]["with spatial features"], rob["cost"]["without spatial features"]
wi, woi = rob["income"]["with spatial features"], rob["income"]["without spatial features"]
story += [
    P("6. Conclusions", h),
    B(f"<b>Tracts can be flagged ahead of time.</b> On unseen tracts the best models reach about {pct(tm('cost', 'XGBoost')['accuracy'])} (cost response) "
      f"and {pct(tm('income', 'L1 Logistic')['accuracy'])} (income response) versus {pct(nir_c)} and {pct(nir_i)} for a no-information guess, with precision and "
      f"recall of roughly 0.6 to 0.7. The cost response is easier to predict than the income-distribution shift. "
      f"However, much of the cost result is mean reversion: tracts with a high 2011 median housing cost saw smaller later increases (r = {au['B_baselines']['corr_level_vs_change_cost']:+.2f}), "
      f"and that one feature alone gives {pct(au['B_baselines']['cost']['1 feature (2011 median cost)'])} in 5-fold CV versus {pct(au['B_baselines']['cost']['all 401 features, XGBoost'])} for the full model."),
    B("<b>Model choice matters little.</b> Tree ensembles and the regularised linear model perform alike (differences are within sampling noise), "
      "suggesting most signal is additive in the 2011 tract characteristics. This differs from the original paper, where non-parametric models clearly beat the logit."),
    B(f"<b>The spatial-disequilibrium features did not help here.</b> Ablation: cost {pct(wc['random_cv_mean'])} with vs {pct(woc['random_cv_mean'])} without "
      f"(county-blocked CV {pct(wc['county_cv_mean'])} vs {pct(woc['county_cv_mean'])}); income {pct(wi['random_cv_mean'])} vs {pct(woi['random_cv_mean'])}. "
      f"Their XGBoost importance ranks are {min(crank.values())} to {max(crank.values())} (cost) and {min(irank.values())} to {max(irank.values())} (income) out of {nfeat}. We therefore do <b>not</b> reproduce the paper's claim that "
      "spatial randomness is the leading indicator. " + why),
    B(f"<b>How much data is enough?</b> We use every California tract (the whole population of about 8,000, not a sample). A learning curve shows the cost response "
      f"is within one point of its best accuracy ({pct(a_c)}) from about {n_c:,} training tracts, and the income response levels off near {pct(a_i)} at about {n_i:,}. "
      f"More California tracts would therefore add little; improving the features or labels matters more than more rows."),
    B("<b>Top predictors (XGBoost gain).</b> Cost response: number of renter-occupied housing units (S2502_C03_001E), number of families earning $75,000 to $99,999 "
      "(DP03_0082E) and the tract's 2011 median monthly housing cost (S2503_C01_028E). Income response: share of owner-occupied households earning $150,000 or more "
      "(S2503_C02_012E), share of renter households earning $75,000+ that spend 30% or more of income on housing (S2503_C03_048E) and owner-occupied households with a "
      "bachelor's degree or higher (S2502_C02_021E). In short, the starting income and housing-cost profile of a tract matters most."),
    B(f"<b>Audit checks.</b> Passed: shuffled labels give {pct(sum(au['C_shuffled']['cost']) / 3)} (cost) and {pct(sum(au['C_shuffled']['income']) / 3)} (income), i.e. chance; "
      f"the median-cost column is consistent across ACS releases (r = {min(au['A1_cost_column_consecutive_corr'].values()):.2f} to {max(au['A1_cost_column_consecutive_corr'].values()):.2f} between consecutive years, no break in 2015); "
      f"the income response correlates {au['A2_income_direction']['pearson']:+.2f} with the change in median household income; "
      f"other windows give {pct(au['E_windows']['features 2010-11, 2012->2015 (3-year horizon)']['cost']['xgb'])} and {pct(au['E_windows']['features 2011-12, 2013->2016 (pivot 2013)']['cost']['xgb'])} (cost), "
      f"{pct(au['E_windows']['features 2010-11, 2012->2015 (3-year horizon)']['income']['xgb'])} and {pct(au['E_windows']['features 2011-12, 2013->2016 (pivot 2013)']['income']['xgb'])} (income). "
      f"Qualifications: with absolute labels (change &gt; 0) the margin over the no-information rate shrinks to "
      f"{100 * (dl['cost']['absolute (change > 0)']['xgb'] - dl['cost']['absolute (change > 0)']['no_info']):.1f} (cost) and {100 * (dl['income']['absolute (change > 0)']['xgb'] - dl['income']['absolute (change > 0)']['no_info']):.1f} (income) points. "
      f"Among initially lower-income tracts (n = {ae['n_eligible']:,}), the classic label (cost and income both rise faster than the median) is ranked with AUC {ae['classic_XGBoost']['auc_eligible_only']:.2f}; "
      f"a watchlist of the top 10% highest-risk tracts is {pct(ae['classic_XGBoost']['precision_top10pct_eligible'])} correct versus a {pct(ae['classic_XGBoost']['base_rate_eligible'])} base rate "
      f"({ae['classic_XGBoost']['lift_top10pct_eligible']:.1f}x lift). The system is a useful ranker, not a precise predictor."),
    B("<b>Caveats.</b> Labels are relative (faster than the median tract), the cost response may partly reflect mean reversion, ACS tract estimates carry sampling error, "
      "and the test set has only 399 tracts, so results are predictive associations, not causal claims, and are not directly comparable with the paper's numbers. "
      "Future work: absolute or displacement-based labels, non-overlapping ACS windows, distance-weighted adjacency, and causal analysis."),
]

SimpleDocTemplate("writeup.pdf", pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.1 * cm, bottomMargin=1.0 * cm,
                  title="Gentrification early-warning system - write-up").build(story)
print("wrote writeup.pdf")
