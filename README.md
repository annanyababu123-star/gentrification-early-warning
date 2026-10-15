# Early-Warning System for Gentrification in California Census Tracts

UE24CS352A Machine Learning mini-project,
Team 40: Annanya Babu (PES1UG24AM045), Aayushman Singh (PES1UG24AM006). 
Based on the approach of Pattabi (Stanford, 2018), re-implemented
on current Census Bureau data (American FactFinder was retired; the same ACS tables are served by the Census API).

**Problem.** Given what a California census tract looked like in 2010-2011, predict whether it will gentrify
between 2012 and 2016. Binary classification, run for two definitions of "gentrification":

| Response | Label = 1 when ... |
|---|---|
| `cost` | the tract's **real median monthly housing cost** rose by more than the median tract (2012 -> 2016, 2016 dollars) |
| `income` | the tract's **income distribution shifted toward affluence** more than the median tract (change in Hellinger distance to an "all households in the top income bin" baseline) |

Thresholds are medians of the *training* set, so classes are ~50/50 and the no-information accuracy is ~50%.

## Data
* American Community Survey 5-year estimates, tract level, California (state 06), releases 2010-2016.
  Tables `S2502`, `S2503` (housing/occupancy by income), `DP03` (economic profile), `B25085`.
* TIGER/Line 2016 tract boundaries (for the queen-contiguity neighbour graph).
* 8,056 tracts appear in every needed year; 7,964 have both responses defined (train 7,167 / validation 398 / test 399).

## Method
1. **Responses** from 2012 vs 2016 (see above). Costs are converted to constant dollars with annual CPI-U.
2. **Features** (observation period 2010-2011): 397 ACS variables from 2011 (after removing columns with >=10% missing,
   constants and near-duplicates with |r|>0.97) + 4 engineered spatial features: spatial lag and Local Moran's I of the
   2010->2011 change in cost and in income distribution, using queen contiguity.
3. **Models**: Random Forest, XGBoost, L1-penalised logistic regression, and a majority-vote ensemble.
   Two-stage grid search (coarse grid, then a refined grid around the best point) scored by 3-fold CV accuracy on the
   training set only. Validation and test sets are never used for tuning.
4. **Evaluation**: accuracy (Wilson 95% CI), precision, recall on the held-out test set vs. the no-information rate
   (exact binomial test). Extra checks: spatial-feature ablation and county-blocked cross-validation.

## Results (test set, n = 399)
| Model | cost accuracy | cost precision / recall | income accuracy | income precision / recall |
|---|---|---|---|---|
| Random Forest | 70.2% | 0.67 / 0.73 | 64.7% | 0.64 / 0.64 |
| XGBoost | 71.7% | 0.69 / 0.74 | 63.4% | 0.62 / 0.63 |
| L1 logistic | 69.7% | 0.67 / 0.72 | 64.9% | 0.65 / 0.61 |
| Ensemble (vote) | 70.9% | 0.68 / 0.74 | 64.9% | 0.64 / 0.63 |
| *No-information rate* | *52.4%* | | *51.1%* | |

All models beat the no-information rate (p < 0.001), but differences **between** models are within sampling noise
(95% CI half-width about 4.5 points). In the ablation, the four engineered spatial features did **not** improve accuracy
(cost: 71.6% with vs 71.2% without, 5-fold CV; income: 63.4% vs 63.6%) and rank low in XGBoost importance. See
`results/` for every number and figure.

## Setup and run
```bash
pip install -r requirements.txt

# 1. get a free API key: https://api.census.gov/data/key_signup.html
export CENSUS_API_KEY=your_key        # Windows cmd: set CENSUS_API_KEY=your_key

# 2. download the data (~2 min) -> ./data
python fetch_census.py

# 3a. quick demo (~3 min, small hyper-parameter grid) -> ./results_fast
FAST=1 python gentrification.py       # Windows cmd: set FAST=1 && python gentrification.py

# 3b. full run (~45 min on 2 cores, full two-stage grid + robustness checks) -> ./results
python gentrification.py

# optional: readable names for Census variable codes, then redraw figures
python fetch_labels.py
python make_figures.py results data
```
The API key is read from an environment variable and must never be committed. `data/` is git-ignored.

## Files
| File | Purpose |
|---|---|
| `fetch_census.py` | downloads ACS tract tables 2010-2016 and the TIGER tract shapefile |
| `gentrification.py` | responses, spatial features, tuning, evaluation, robustness checks |
| `make_figures.py` | figures from saved results (no re-training) |
| `fetch_labels.py` | human-readable names for Census variable codes |
| `learning_curve.py` | accuracy vs. number of training tracts (answers "how much data is enough?", ~3 min) |
| `diagnose_spatial.py` | why the spatial features did not help (pre/post change correlation, label clustering, single-feature AUCs, neighbour-level features; ~5 min) |
| `audit.py`, `audit_eligible.py` | pipeline audit: response validity, mean-reversion baselines, shuffled labels, label/window sensitivity, error analysis, classic-gentrification view (~10 min and ~3 min) |
| `build_writeup.py` | builds the 2-page PDF write-up from `results/` |
| `results/` | metrics (`metrics_*`, `results.json`), tuning logs, feature importances, figures |

## Why the spatial features did not help
`diagnose_spatial.py` (output in `results/diagnose_spatial.json`): the 2010-2011 changes that feed the spatial lag / Local
Moran features barely predict the 2012-2016 change (correlation -0.05 cost, -0.17 income; ACS windows overlap), the 2011
levels already carry the signal (single-feature AUC 0.69 for median housing cost vs <= 0.58 for any spatial feature),
and the labels are only moderately clustered (Moran's I 0.14 cost, 0.04 income). Neighbour averages of the top-10 level
features add only about +0.2 to +1.0 accuracy points, comparable to fold-to-fold variation.

## Audit (what we checked, and what it means)
`audit.py` and `audit_eligible.py` write `results/audit.json` and `results/audit_eligible.json`.
* **Passed:** shuffled labels give 50% (no leakage); the median-cost column is consistent across ACS releases
  (r = 0.98-0.99 between consecutive years, no break in 2015); the income response correlates +0.45 with the change in
  median household income; accuracy holds for other windows (cost 67-70%, income 62-63%) and by tract size and county.
* **Qualification 1, mean reversion:** tracts with a high 2011 median cost grew less (r = -0.45). That single feature gives
  64.1% on the cost response (5-fold CV); the full 401-feature model gives 71.6%.
* **Qualification 2, label choice:** with absolute labels (change > 0) the margin over the no-information rate is only
  about +6.6 points (cost) and +3.4 points (income).
* **Qualification 3, classic definition:** among initially lower-income tracts (n = 3,982) the combined label (cost and
  income both rise faster than the median) is ranked with AUC 0.67; a top-10% watchlist is 55.8% correct vs a 30.5%
  base rate (1.8x lift). Useful as a ranker, not a precise predictor.

## How much data is enough?
We use every California tract (the whole population of about 8,000 tracts, not a sample). `learning_curve.py` trains on
random subsets of the training tracts and scores on 797 held-out tracts: the cost response is within one point of its
best accuracy from about 1,400 training tracts, and the income response levels off at about 5,000. More California
tracts would add little; better features or labels would matter more (`results/fig_learning_curve.png`).

## Limitations (please read)
* ACS 5-year estimates are rolling windows (2011 covers 2007-2011, 2012 covers 2008-2012, ...). The observation and
  prediction windows overlap, and tract-level estimates carry sampling error. Predictive accuracy is therefore not a
  causal statement about gentrification.
* The labels are *relative* (faster-than-median change), not an absolute definition of gentrification, and the cost
  response can partly reflect mean reversion of high-cost tracts. Numbers are **not** directly comparable with the
  original paper, which used different label thresholds.
* The Random Forest's best `max_features` sits at the edge of its grid; a wider grid might help slightly.
* One random train/validation/test split; the test set has 399 tracts, hence the wide confidence intervals.
