# Time to churn

> Part of the [ChurnGuard README](../README.md). This page holds the detail; every number here is the same as in the README.

![Kaplan-Meier survival curves](../reports/figures/survival_km.png)

The classifier answers *whether* a customer looks like a churner. Survival analysis answers *when*,
using tenure as the clock ([`survival.py`](../src/churnguard/survival.py); full write-up in
[`reports/survival_summary.md`](../reports/survival_summary.md)). Customers who haven't left are
*censored*: we only know they lasted at least their tenure.

**Hazard vs probability.** A *probability* says how likely a customer is flagged as churned. A
*hazard* is the rate of leaving at a given tenure among customers still active then, so a hazard ratio
of 2 means "leaves at twice the rate at any given tenure", not "twice as likely to ever leave".

**Key finding: the contract is the main time-to-churn split.** Of month-to-month customers, 70.3% are
still active at 12 months and 49.1% at 36 (median tenure at churn: 35 months). For one-year contracts
the figures are 99.1% and 95.9%; for two-year, 100.0% and 99.9%. Fiber-optic customers fall faster than
DSL (78.4% vs 86.8% active at 12 months). In a Cox model fit on the training split, holding the other
covariates fixed, a two-year contract has a hazard ratio of 0.07 and a one-year contract 0.22 relative to
month-to-month; fiber optic is 1.58 relative to DSL and paying by electronic check is 1.94.

- **Concordance 0.855 on validation** (0.861 on train; 0.5 is chance). It measures how well the model
  orders who leaves first, which is a different question from ROC-AUC, so the two numbers are not
  comparable.
- **Proportional hazards partly fails.** The Schoenfeld test flags 2 of 13 covariates at p < 0.01:
  `Contract_One year` (p = 0.0002) and `n_addon_services` (p = 0.0002). For those, one hazard ratio is
  an average over an effect that changes with tenure. The one-year curve declines slowly until about 50 months
  and then drops steeply (to roughly 57% by month 72), which is what a changing effect looks like. Read
  those two ratios as summaries; the Kaplan-Meier curves do not assume proportionality.
- `TotalCharges` is left out of the Cox model because it is about tenure × monthly charge and would
  leak the clock into the covariates. The 11 tenure-0 customers are dropped (no time at risk).
- These are associations in observational data, not effects of changing a contract.

```bash
pip install -c constraints.txt ".[analysis]"   # lifelines; not needed to serve the API
python -m churnguard.survival                  # writes reports/survival_summary.{md,json} and figures
```
