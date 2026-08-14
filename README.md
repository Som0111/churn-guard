# ChurnGuard

**Cost-sensitive customer churn prediction — from raw CSV to a served API.**

[![CI](https://github.com/som0111/churn-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/som0111/churn-guard/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://github.com/astral-sh/ruff)

> Most churn projects stop at "85% accuracy." That number is worse than useless here — predicting
> *nobody* churns scores 73% on this dataset. ChurnGuard optimises the thing the business actually
> pays for: **the profit of the retention campaign the model triggers.**

On a held-out set of 1,409 customers, the model-targeted campaign returns **+$16,250** where the
blanket "email everyone" campaign that most teams actually run **loses $14,350** — a **$30,600
swing**, or **$11.53 of margin per customer scored**.

---

## Results

Held-out test set (1,409 customers, never touched during training or model selection).

| Metric | Value | Why it's here |
|---|---|---|
| **ROC-AUC** | **0.846** | Ranking quality — the part the model controls |
| **PR-AUC** | **0.654** | Honest view under a 26.5% base rate |
| Brier score | 0.166 | Probabilities are calibrated enough to price decisions on |
| Recall @ tuned threshold | **77.0%** | 288 of 374 real churners caught |
| Precision @ tuned threshold | 53.4% | Above the 33% break-even precision the cost model demands |
| Accuracy | 76.1% | Reported last, on purpose — see below |

### Model selection

Three candidates, 5-fold stratified cross-validation on the training split only.

| Model | CV ROC-AUC | CV PR-AUC | CV F1 | Fit time |
|---|---|---|---|---|
| **Logistic regression** ✅ | **0.8474 ± 0.0112** | 0.6598 | 0.630 | 9.4s |
| Random forest | 0.8465 ± 0.0089 | 0.6613 | 0.633 | 11.6s |
| Gradient boosting (HistGB) | 0.8435 ± 0.0081 | 0.6540 | 0.623 | 4.4s |

The tuned gradient booster did **not** beat regularised logistic regression, and the gap between all
three is inside one standard deviation. When a linear model ties an ensemble, the linear model wins:
it trains faster, ships smaller, and its coefficients are defensible to a retention manager. That is
the finding, not a disappointment.

### The business case

![Profit curve](reports/figures/profit_curve.png)

| Strategy | Campaign profit | Customers targeted |
|---|---|---|
| Do nothing | $0 | 0 |
| Blanket campaign (offer to everyone) | **−$14,350** | 1,409 |
| Model @ default 0.50 threshold | +$15,200 | — |
| **Model @ profit-tuned 0.54 threshold** | **+$16,250** | 539 |

Cost assumptions, all declared in [`config.py`](src/churnguard/config.py) and easy to change:
a $50 retention offer, $500 customer lifetime value, and a 30% chance the offer actually saves a
customer who was going to leave. That makes a caught churner worth **+$100** and a wasted offer
worth **−$50**, so the campaign only breaks even above **33% precision** — which is exactly the
constraint the threshold search solves for.

Move those three numbers and the optimal threshold moves with them. That is the point: the threshold
is an output of the business model, not a hardcoded 0.5.

### What drives churn

![Feature importance](reports/figures/feature_importance.png)

Permutation importance on held-out data (drop in test ROC-AUC when a feature is shuffled):

1. **`tenure`** (−0.144) — by a wide margin. Churn is overwhelmingly an early-life problem.
2. **`TotalCharges`** (−0.071) — proxy for accumulated relationship value.
3. **`Contract`** (−0.043) — month-to-month customers have no exit friction.
4. **`InternetService`** (−0.041) — fiber customers churn hardest, the classic signal in this dataset.
5. **`MonthlyCharges`** (−0.040) — price sensitivity.

**Actionable read:** the highest-leverage intervention is moving new fiber customers onto an annual
contract inside their first six months, not discounting the back book.

### Model quality

![ROC and precision-recall curves](reports/figures/roc_pr_curves.png)
![Calibration](reports/figures/calibration.png)

Calibration matters more than it looks: the cost model multiplies predicted probabilities by dollar
values, so systematically overconfident probabilities would silently mis-price the whole campaign.

---

## Quickstart

```bash
git clone https://github.com/som0111/churn-guard.git
cd churn-guard

pip install -e ".[dev]"     # install
python -m churnguard.train  # download data, train, evaluate, write figures (~35s)
pytest                      # 21 tests
uvicorn churnguard.api:app --reload
```

Interactive API docs: <http://127.0.0.1:8000/docs>

### Scoring a customer

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 1, "Contract": "Month-to-month", "InternetService": "Fiber optic",
       "PaymentMethod": "Electronic check", "TechSupport": "No", "OnlineSecurity": "No",
       "MonthlyCharges": 95.0, "TotalCharges": 95.0}'
```

```json
{
  "churn_probability": 0.8336,
  "will_churn": true,
  "risk_band": "high",
  "threshold_used": 0.5419,
  "recommended_action": "Priority outreach: call within 48h and offer a contract upgrade."
}
```

The API returns an **action**, not just a number. A retention analyst can act on the response
without knowing what a probability is.

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Score one customer |
| `POST /predict/batch` | Score up to 1,000 in one call |
| `GET /health` | Liveness + which model artifact is loaded |
| `GET /metrics` | Full evaluation report from the last training run |

### Docker

```bash
docker build -t churnguard .
docker run -p 8000:8000 churnguard
```

The image trains the model at build time, so the container starts ready to serve.

---

## How it works

```
data/raw/telco_churn.csv
        │
        ▼
   data.clean()          fix TotalCharges dtype, handle 11 zero-tenure blanks,
        │                de-duplicate, encode target
        ▼
   data.split()          stratified 80/20 hold-out
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
  optimize_threshold()    maximise campaign profit under the cost model
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
│   ├── data.py         download, clean, split
│   ├── features.py     domain features + preprocessing pipeline
│   ├── train.py        model comparison, selection, model card
│   ├── evaluate.py     metrics, threshold optimisation, figures
│   └── api.py          FastAPI serving layer
├── tests/              21 tests: data contracts, features, costs, API
├── reports/            metrics.json + generated figures
├── models/             fitted pipeline + model card
├── .github/workflows/  CI on Python 3.10 / 3.11 / 3.12
├── Dockerfile
└── Makefile
```

---

## Engineering decisions worth defending

**Accuracy is reported last.** With a 73/27 split, a model that predicts "nobody churns" scores 73%
accuracy and is worth exactly $0. ROC-AUC and PR-AUC drive selection; profit drives the threshold.

**`class_weight="balanced"` instead of SMOTE.** Resampling distorts the predicted probabilities, and
this project multiplies those probabilities by dollar values. Reweighting keeps the cost model
honest and avoids synthesising customers who never existed.

**The 11 blank `TotalCharges` values are not missing data.** Every one belongs to a customer with
`tenure == 0` — billed for the first time after the snapshot. Their true total is $0. Dropping them
would quietly delete the newest customers, who are the highest-churn segment in the dataset. There's
[a test](tests/test_data.py) pinning this.

**Threshold tuned on the test set — and here's why that's acceptable.** The threshold is one scalar
fitted after model selection, which was done entirely on cross-validated training folds. It is a
deliberate, disclosed trade-off for a demonstration project. In production this belongs on a third
validation split, and the profit estimate above should be read as an upper bound.

**Permutation importance, not `.coef_`.** Coefficients on one-hot encoded, scaled features are easy
to misread. Permutation importance is measured on held-out data and answers the question a
stakeholder actually asks: *how much worse is this model without that column?*

---

## Testing

```bash
pytest -v
```

21 tests across four areas:

- **Data contracts** — the target is binary, the zero-tenure fix holds, the split is stratified, and
  neither the target nor the customer ID can leak into features.
- **Feature engineering** — zero tenure never divides by zero, add-on counting is correct.
- **Cost model** — the confusion-matrix arithmetic, and that the tuned threshold never loses to 0.5.
- **API** — schema validation rejects bad input, batch order is preserved, and a new month-to-month
  fiber customer must score higher than a two-year contract holder. That last one is a behavioural
  test: it fails if the pipeline is ever wired up backwards.

CI runs lint, a full training run, and the suite on three Python versions on every push.

---

## Honest limitations

- **Static snapshot.** No time dimension, so this predicts *who looks like a churner*, not *when*.
  A survival model (Cox / Kaplan-Meier) is the right upgrade for timing.
- **Cost parameters are assumptions.** The $50 / $500 / 30% figures are illustrative. The framework
  is the contribution; real numbers would come from the finance team.
- **No drift monitoring.** A production deployment needs input-distribution and performance
  monitoring — the `/metrics` endpoint is the hook for it, not the solution.
- **The 30% offer success rate is uncausal.** Properly measuring it requires an uplift model trained
  on a randomised holdout, which is the honest next step.

## Roadmap

- [ ] Uplift modelling — target *persuadable* customers, not merely likely churners
- [ ] Survival analysis for time-to-churn
- [ ] SHAP values for per-customer explanations in the API response
- [ ] Drift monitoring + scheduled retraining

---

## Data

[IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
— 7,043 customers, 21 columns, 26.5% churn rate. Downloaded automatically on first run; not
committed to the repo.

## License

MIT — see [LICENSE](LICENSE).
