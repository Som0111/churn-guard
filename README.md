# ChurnGuard

**Cost-sensitive customer churn prediction — from raw CSV to a served API.**

[![CI](https://github.com/som0111/churn-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/som0111/churn-guard/actions/workflows/ci.yml) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![Code style: ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://github.com/astral-sh/ruff)

**[▶ Try the live API](https://churn-guard-api.onrender.com/docs)** — score a customer in your browser.
*(Free tier: an idle instance sleeps and the first request after it can take most of a minute to wake — 43s
was observed on 2026-10-02; a warm request took under a second. This public endpoint is an
**unauthenticated demo**: no API key, no rate limit — do not send real customer data to it.)*

> Most churn projects stop at "85% accuracy." That number is worse than useless here — predicting
> *nobody* churns scores 73% on this dataset. ChurnGuard optimises the thing the business actually
> pays for: **the profit of the retention campaign the model triggers.**

```json
{"churn_probability": 0.7092, "will_churn": true, "risk_band": "high", "threshold_used": 0.3837,
 "recommended_action": "Priority outreach: call within 48h and offer a contract upgrade.",
 "top_drivers": [{"feature": "tenure", "direction": "raises", "magnitude": 1.7518},
                 {"feature": "Contract", "direction": "raises", "magnitude": 0.6835},
                 {"feature": "InternetService", "direction": "raises", "magnitude": 0.6016}]}
```

| On a held-out test set of 1,409 customers | |
|---|---|
| Model-targeted campaign, projected profit | **+$15,900** (90% interval $13,000 to $18,600) |
| Blanket "email everyone" campaign | **−$14,350** (interval −$18,850 to −$10,450) — a **$30,250 swing**, **$11.28 per customer scored** |
| ROC-AUC | **0.846** |
| Calibration (ECE) | **0.020** |
| Recall / precision at the tuned threshold | **67.7%** / 57.4% |

*Projections under assumed costs ($50 offer, $500 customer value, 30% save rate), not observed outcomes —
see [what this project does not prove](#what-this-project-does-not-prove). The threshold was chosen on a separate validation split, not on the test set.*

![Profit curve](reports/figures/profit_curve.png)

```bash
pip install -c constraints.txt -e ".[dev,explain]"   # pinned install (Python 3.12; SHAP adds ~150 MB)
python -m churnguard.train                           # download data, train, evaluate (~35s)
uvicorn churnguard.api:app --reload                  # http://127.0.0.1:8000/docs
```

More: [full quickstart, endpoints, Docker and deployment](docs/api-and-deployment.md) ·
[CHANGELOG](CHANGELOG.md) (before/after numbers for every fix) ·
[docs/DECISIONS.md](docs/DECISIONS.md) (why each choice was made).

---

## How it was evaluated

![Sensitivity heatmap](reports/figures/sensitivity_heatmap.png)

Data is split 60/20/20 (train / validation / test): train selects the model, validation picks the profit
threshold, and test is scored once. The campaign loses money in 45 of 144 cost/save-rate combinations, but beats
the blanket campaign in 140 of 144 (heatmap: profit at the shipped threshold across save rate 10–50% and offer cost $25–$100).
→ [Methodology and results](docs/methodology.md): protocol, bootstrap intervals, model selection, business case, decisions.

## Calibration

![Calibration](reports/figures/calibration.png)

`class_weight="balanced"` predicted a 40.7% mean churn rate for a 26.5% population (ECE 0.142); the unweighted model
fixes that (ECE 0.019 on validation, 0.020 on test) and the threshold moved from 0.64 to 0.38.
→ [Calibration audit](docs/calibration.md)

## What the model relies on

![What the model relies on](reports/figures/feature_importance.png)

`tenure` + `TotalCharges` (shuffled together) cost 0.110 AUC, then `Contract` 0.051, `MonthlyCharges` 0.033 and
`InternetService` 0.023 — associations the model uses, not causes. `/predict` also returns per-customer SHAP `top_drivers`.
→ [Importance and SHAP drivers](docs/explainability.md) (including why `tenure` alone first read 0.375)

## Time to churn

![Kaplan-Meier survival curves](reports/figures/survival_km.png)

Month-to-month customers are 70.3% still active at 12 months; one-year contracts 99.1% and two-year 100.0%. A Cox model
reaches 0.855 concordance on validation, though proportional hazards fails for 2 of 13 covariates.
→ [Survival analysis](docs/survival.md) (optional `.[analysis]` extra; report only)

## Monitoring

![Drift demo](reports/figures/drift_demo.png)

`POST /drift` compares a recent batch to a validation-split reference with KS and chi-square tests. In the demo a clean
batch is `ok` (0 of 19 fields drifted) and a simulated shift is `alert` (6 of 19) — a narrow margin over the 30% line.
→ [Drift monitoring](docs/monitoring.md) (Evidently is an optional extra, only for the HTML report)

## API, Docker and testing

`/predict`, `/predict/batch`, `/health/live`, `/health/ready`, `/model-info`, `/metrics` and `/drift`; non-root Docker image
pinned by digest, built and smoke-tested in CI; live instance checked on 2026-10-02. 154 tests (80.5% coverage, CI floor 79%).
→ [API, Docker and deployment](docs/api-and-deployment.md) · [Testing](docs/testing.md) ·
[Glossary](docs/glossary.md) (ECE, hazard ratio, SHAP, censored, …)

---

## What this project does not prove

- **That the offers work.** The 30% save rate is an assumption. Nothing here measures whether an offer changes
  anyone's decision; that takes a randomised test (hold some at-risk customers out, compare outcomes).
- **That the campaign would make money.** The +$15,900 is the model's targeting plus assumed costs. The
  [sensitivity grid](docs/methodology.md#how-sensitive-is-this-to-the-assumptions) shows the campaign *losing* money in 45 of 144
  cost/save-rate combinations. Real ROI needs real offer costs, real save rates and real margins.
- **That the "drivers" are causes.** Permutation importance, SHAP values and hazard ratios describe what the
  model uses and what co-occurs with churn in one observational dataset. They do not show that moving a
  customer onto an annual contract would keep them.
- **That the model holds up in production.** The drift demo is a simulation I designed, its thresholds are
  uncalibrated, and the model has never seen live traffic or confirmed outcomes.
- **That it generalises.** One public telecom dataset, one snapshot, one random split.

## Honest limitations

- **Static snapshot.** The classifier predicts *who looks like a churner*, not *when*. The
  [time-to-churn analysis](docs/survival.md#time-to-churn) addresses timing, but only as a report: it is not served
  from the API, and tenure is a single snapshot, not a customer history.
- **Cost parameters are assumptions.** The $50 / $500 / 30% figures are illustrative. The framework
  is the contribution; real numbers would come from the finance team.
- **One public dataset.** IBM's Telco sample (7,043 customers, one snapshot). Nothing here shows the model
  transfers to another company, market or period.
- **Drift monitoring covers inputs and predictions only.** `/drift` cannot see model *performance*
  decay: that needs the true churn outcomes, which arrive weeks later, and no part of this project
  collects them. Nothing is scheduled; someone has to call `/drift`.
- **The live demo has no authentication and no rate limit.**
- **Fairness and segment-level performance were not examined.** Overall metrics can hide a model that works
  worse for some customer groups.
- **Validation was reused.** It picks the calibration variant and tunes the threshold, so the validation figures
  carry a small selection effect; the test figures do not.

## Roadmap

- [ ] Uplift modelling — target *persuadable* customers, not merely likely churners (needs a randomised offer test)
- [x] Time-to-churn analysis (Kaplan-Meier + Cox; report only)
- [x] Per-customer SHAP drivers in the API response (optional extra; active on the live demo)
- [x] Drift monitoring for inputs and predictions
- [ ] Outcome-based performance monitoring, scheduled drift checks and retraining

## License

MIT — see [LICENSE](LICENSE).
