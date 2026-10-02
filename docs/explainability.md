# What drives the predictions

> Part of the [ChurnGuard README](../README.md). This page holds the detail; every number here is the same as in the README.

## What the model relies on

![What the model relies on](../reports/figures/feature_importance.png)

Permutation importance on held-out data: how much test ROC-AUC the model **loses** when a field is shuffled.
This describes the *model*, not the world — a field the model leans on is **strongly associated with higher
predicted churn**, which is not evidence that changing it would change anyone's behaviour.

| Field | AUC lost when shuffled |
|---|---|
| **`tenure` + `TotalCharges`** (shuffled together) | **0.110** |
| `Contract` | 0.051 |
| `MonthlyCharges` | 0.033 |
| `InternetService` | 0.023 |
| every other field | ≤ 0.003 each |

- **Shorter tenure and month-to-month contracts are strongly associated with higher predicted churn**, and
  fiber-optic customers have higher observed churn than DSL customers in this dataset. The survival analysis
  below shows the same pattern in time-to-churn terms.
- **`tenure` and `TotalCharges` are shuffled as one block on purpose.** An earlier version shuffled `tenure`
  alone and reported a drop of 0.375 — more than the model's whole headroom above a coin flip (0.846 − 0.5 =
  0.346). Cause: `avg_monthly_spend` is `TotalCharges ÷ tenure`, so shuffling tenure against a fixed
  `TotalCharges` manufactures impossible customers (that feature's 99th percentile goes from $115 to
  $6,327; `spend_vs_current_ratio` from 1.2 to 68.0), and the model's score on them is worse than random
  (AUC 0.452). The 0.375 measured how badly the model copes with nonsense rows, not how much it needs tenure.
  Shuffling the five tenure-derived features together in the model's own input space gives 0.114, in line with
  the joint 0.110. Reproduce with
  [`scripts/investigate_permutation_importance.py`](../scripts/investigate_permutation_importance.py).
- Importance is split across redundant features: `tenure`, `tenure_years` and `is_new_customer` lose only
  ~0.017 each when shuffled alone, because the others still carry the signal.
- `spend_vs_current_ratio` is this month's charge divided by the customer's *lifetime average* charge. It is
  **not** a measured price rise — the dataset has no billing history.

## Why this score? (SHAP drivers)

`top_drivers` lists the three customer fields moving this score most, from per-customer
[SHAP](https://github.com/shap/shap) values (`LinearExplainer` for logistic regression,
`TreeExplainer` for random forests). `direction` says whether the field raises or lowers the churn
risk relative to an average training customer; `magnitude` is the absolute contribution in log-odds.

**Grouping.** SHAP runs on the model's transformed inputs, so contributions are folded back onto the
original customer fields before ranking, so that drivers read as fields the analyst recognises:

- one-hot columns go to their field (`Contract_Two year` → `Contract`);
- engineered features go to the field they came from (`tenure_years` and `is_new_customer` → `tenure`);
- a feature built from several fields is split equally between them (`avg_monthly_spend` →
  half `TotalCharges`, half `tenure`; `spend_vs_current_ratio` → thirds across `MonthlyCharges`,
  `TotalCharges`, `tenure`; `n_addon_services` → sixths across the six add-on services).

Grouping only re-labels: the grouped values still sum to the model's total contribution (tested).
The equal split is a convention, not a measurement of which source "really" mattered. These are
associations in the model, not causes: a driver that "raises" risk is not shown to be something that,
if changed, would change the customer's behaviour. `shap` is an optional extra (`.[explain]`,
about +150 MB of numba/llvmlite); without it the API still scores and `top_drivers` is `null`.
Measured on one Windows machine (in-process, 300 requests): `/predict` p50 latency 19.7 ms without
drivers, 35.0 ms with.
