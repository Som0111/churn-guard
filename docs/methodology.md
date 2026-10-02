# Methodology and results

> Part of the [ChurnGuard README](../README.md). This page holds the detail; every number here is the same as in the README.

## Headline result

On a held-out test set of 1,409 customers, a model-targeted retention campaign is projected to net
**+$15,900 (90% interval $13,000 to $18,600)**, where the blanket "email everyone" campaign **loses $14,350**
(interval −$18,850 to −$10,450) — a **$30,250 swing**, or **$11.28 of margin per customer scored**. These are
*projections under assumed costs*, not observed outcomes: a $50 offer, $500 customer value and a 30% save rate
(see [sensitivity](methodology.md#how-sensitive-is-this-to-the-assumptions) and
[what this project does not prove](../README.md#what-this-project-does-not-prove)). The threshold behind that number was
chosen on a separate validation split, not on the test set.

## Results and evaluation protocol

Held-out test set (1,409 customers). Data is split 60/20/20 (4,225 train / 1,409 validation /
1,409 test, stratified): **train** is for model selection (cross-validation), **validation** is for
choosing the decision threshold, and **test** is scored once with that threshold frozen.

| Metric | Value | Why it's here |
|---|---|---|
| **ROC-AUC** | **0.846** | Ranking quality — the part the model controls |
| **PR-AUC** | **0.654** | Honest view under a 26.5% base rate |
| Brier score | 0.136 | Constant base-rate predictor scores 0.195 (lower is better) |
| Expected calibration error (ECE) | 0.020 | Mean gap between predicted and observed churn rate — see [Calibration](calibration.md#calibration) |
| Recall @ tuned threshold | **67.7%** | 253 of 374 real churners caught |
| Precision @ tuned threshold | 57.4% | Above the 33% break-even precision the cost model demands |
| Accuracy | 78.1% | Reported last, on purpose — see below |

## How certain is it?

Bootstrap of the 1,409 test customers (1,000 seeded resamples, 90% percentile intervals) at the
frozen 0.38 threshold:

| Quantity | Point estimate | 90% interval |
|---|---|---|
| Model campaign profit | +$15,900 | $13,000 to $18,600 |
| Blanket campaign profit | −$14,350 | −$18,850 to −$10,450 |
| Do nothing | $0 | $0 |
| ROC-AUC | 0.846 | 0.828 to 0.864 |
| PR-AUC | 0.655 | 0.610 to 0.696 |
| Precision | 57.4% | 53.3% to 61.2% |
| Recall | 67.7% | 63.7% to 71.6% |

This captures sampling noise in the test customers only. It does not include uncertainty in how the
threshold was chosen, in the model fit, or — the biggest one — in the cost assumptions below.

## Model selection

Three candidates, 5-fold stratified cross-validation on the training split only.

| Model | CV ROC-AUC | CV PR-AUC | CV F1 | Fit time |
|---|---|---|---|---|
| **Logistic regression** ✅ | **0.8501 ± 0.0135** | 0.6659 | 0.633 | 6.8s |
| Random forest | 0.8482 ± 0.0116 | 0.6713 | 0.633 | 8.6s |
| Gradient boosting (HistGB) | 0.8419 ± 0.0111 | 0.6578 | 0.624 | 2.9s |

The tuned gradient booster did **not** beat regularised logistic regression, and the gap between all
three is inside one standard deviation. When a linear model ties an ensemble, the linear model wins:
it trains faster, ships smaller, and its coefficients are defensible to a retention manager. That is
the finding, not a disappointment.

## The business case

![Profit curve](../reports/figures/profit_curve.png)

| Strategy | Campaign profit | Customers targeted |
|---|---|---|
| Do nothing | $0 | 0 |
| Blanket campaign (offer to everyone) | **−$14,350** | 1,409 |
| Model @ default 0.50 threshold | +$15,050 | — |
| **Model @ 0.38 threshold (tuned on validation)** | **+$15,900** | 441 |

On the validation split the same threshold earns $16,950, so the gap to the test figure is the
honest cost of tuning on one sample and scoring on another. All rows above are test-set numbers.

**Assumptions, not observed outcomes.** Cost assumptions, all declared in [`config.py`](../src/churnguard/config.py) and easy to change:
a $50 retention offer, $500 customer lifetime value, and a 30% chance the offer actually saves a
customer who was going to leave. That makes a caught churner worth **+$100** and a wasted offer
worth **−$50**, so the campaign only breaks even above **33% precision** — which is exactly the
constraint the threshold search solves for.

Move those three numbers and the optimal threshold moves with them. That is the point: the threshold
is an output of the business model, not a hardcoded 0.5.

## How sensitive is this to the assumptions?

![Sensitivity heatmap](../reports/figures/sensitivity_heatmap.png)

The **$50 offer / $500 value / 30% save rate are assumptions, not measured results.** The grid above
re-prices the *shipped* threshold (0.38, not re-tuned) across save rate 10–50% and offer cost $25–$100.

- **The campaign loses money in 45 of 144 cells** — when the save rate is low relative to the offer
  cost. At $50 per offer it needs about a 20% save rate to turn a profit (−$3,075 at 15%, +$3,250 at
  20%); at $100 it needs about 35%. If the real save rate is 10%, this campaign loses $9,400 at $50.
- **The model beats the blanket campaign in 140 of 144 cells.** Blanket mailing only ties or wins when
  offers are very cheap and very effective ($25 at 40%+ save rate, $30 at 50%). Outside that corner,
  targeting beats the blanket campaign everywhere on this grid.

## Model quality

![ROC and precision-recall curves](../reports/figures/roc_pr_curves.png)
![Calibration](../reports/figures/calibration.png)

## How it works

```
data/raw/telco_churn.csv
        │
        ▼
   data.clean()          fix TotalCharges dtype, handle 11 zero-tenure blanks,
        │                de-duplicate, encode target
        ▼
   data.split()          stratified 60/20/20 train / validation / test
        │
        ▼
┌──────────────────────────── sklearn Pipeline ────────────────────────────┐
│  add_domain_features()   avg_monthly_spend, spend_vs_current_ratio,      │
│         │                n_addon_services, tenure_years, is_new_customer │
│         ▼                                                                │
│  ColumnTransformer       median impute + scale  |  mode impute + one-hot  │
│         │                                                                │
│         ▼                                                                │
│  Estimator               selected by 5-fold CV ROC-AUC                    │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  optimize_threshold()    maximise campaign profit on the VALIDATION split;
        │                 test is then scored once at that frozen threshold
        │
        ▼
  churn_pipeline.joblib ──────► FastAPI /predict
```

**Everything lives inside one `Pipeline` object.** Feature engineering, imputation and encoding are
fitted stages, not a notebook cell that ran once. The API loads that exact object, so training and
serving cannot drift apart — the single most common way a working model breaks in production.

### Project structure

```
churn-guard/
├── src/churnguard/
│   ├── config.py       paths, schema, and the business cost model
│   ├── data.py         download, clean, 60/20/20 split
│   ├── features.py     domain features + preprocessing pipeline
│   ├── train.py        model comparison, selection, model card
│   ├── evaluate.py     metrics, threshold optimisation, figures
│   ├── explain.py      SHAP drivers grouped to original fields
│   ├── drift.py        drift detection, demo, optional Evidently HTML report
│   ├── survival.py     Kaplan-Meier + Cox time-to-churn analysis (optional extra)
│   └── api.py          FastAPI serving layer
├── tests/              154 tests: data, validation, provenance, features, costs, API, drift, survival, training
├── reports/            metrics.json + generated figures
├── models/             fitted pipeline + model card
├── .github/workflows/  CI on Python 3.12 (production) and 3.11 (compatibility)
├── render.yaml         free-tier deployment blueprint
├── constraints.txt     pinned dependency versions (generated from pyproject.toml)
├── scripts/            smoke_test.sh (CI container job), investigate_permutation_importance.py
├── docs/               DECISIONS.md — one entry per fix: what, why, trade-off
├── CHANGELOG.md        before/after numbers for every fix
├── Dockerfile
└── Makefile
```

## Engineering decisions worth defending

**Accuracy is reported last.** With a 73/27 split, a model that predicts "nobody churns" scores 73%
accuracy and is worth exactly $0. ROC-AUC and PR-AUC drive selection; profit drives the threshold.

**No resampling and no class weights.** SMOTE distorts predicted probabilities, and this project
multiplies those probabilities by dollar values. `class_weight="balanced"` distorts them too: it
was the original choice, and the calibration audit showed it inflated the mean predicted churn rate
from 26.5% to 40.7% (ECE 0.142). The final model is trained unweighted (ECE 0.019); the profit
threshold, not the loss function, is what handles the class imbalance.

**The 11 blank `TotalCharges` values are not missing data.** Every one belongs to a customer with
`tenure == 0` — billed for the first time after the snapshot. Their true total is $0. Dropping them
would quietly delete the newest customers, who are the highest-churn segment in the dataset. There's
[a test](../tests/test_data.py) pinning this.

**Three splits, three jobs.** Train is used for model selection (5-fold CV) only. The profit
threshold is tuned on a separate validation split, searching every unique predicted probability (plus
0 and 1) rather than a fixed grid. The test split is scored once at that frozen threshold, so the
headline profit never saw test labels when its threshold was picked. The first version of this
project tuned the threshold on the test set; fixing that moved the headline from +$16,250 to +$15,900.

**Permutation importance, not `.coef_` — with linked fields shuffled together.** Coefficients on one-hot
encoded, scaled features are easy to misread, and permutation importance is measured on held-out data. But a
naive per-column shuffle is wrong when features are computed from each other: it reported `tenure` as costing
0.375 AUC, which is impossible (see [What the model relies on](explainability.md#what-the-model-relies-on)). `tenure` and
`TotalCharges` are now shuffled as one block. It is still a statement about the model, not about causes.

## Input validation

Bad data or config fails early instead of training on garbage:

- **Raw data** is checked right after load (`data.validate_raw`): required columns, non-null customer
  ID, target in `{Yes, No}`, every categorical within its allowed set, numeric columns parseable and
  in range. One error lists every problem found. Blank `TotalCharges` is still accepted — those are the
  documented tenure-0 customers.
- **`CostModel`** raises `ValueError` for a negative offer cost or lifetime value, or a success rate
  outside `[0, 1]` (NaN included).
- **API** requests are bounded (tenure 0–100, `MonthlyCharges` ≤ 1,000, `TotalCharges` ≤ 100,000,
  no NaN/Infinity) and carry one conservative consistency check: a customer with tenure 0 cannot have
  `TotalCharges` above one month's charge. 422 responses list the field and message only; they do not
  echo the rejected input back.

## Data

[IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
— 7,043 customers, 21 columns, 26.5% churn rate. Downloaded automatically on first run; not
committed to the repo. The file is verified against a pinned SHA-256 on download and on every load
(`16320c9c…e91`, 970,457 bytes); a mismatch raises `DatasetIntegrityError` instead of silently
training on different data.

## Model provenance

Every training run writes a `provenance` block to `reports/metrics.json` (served by `/metrics`):
`model_version` (`<git sha>-<UTC timestamp>`), git SHA and whether the code tree was dirty (captured before training writes anything, and ignoring `reports/` and
`models/`, so it is `true` only if source files differed from the commit; `null` when there is no `.git`), dataset URL and
SHA-256, selected estimator and its hyperparameters, calibration and threshold methods, NumPy /
pandas / scikit-learn / SciPy / joblib / Python versions, and train / validation / test counts with
class rates. The committed report was produced on Python 3.11.9 with the pinned versions; the same
metrics came out of an earlier run on Python 3.14 with unpinned latest libraries, so the results
are not sensitive to these versions on this machine. They have not been re-run on 3.12 or Linux.
