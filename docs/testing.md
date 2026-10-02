# Testing

> Part of the [ChurnGuard README](../README.md). This page holds the detail; every number here is the same as in the README.

```bash
pytest -v                                   # everything (~3 min)
pytest -m "not integration"                  # skip the full training run (~15s)
pytest --cov=churnguard --cov-report=term-missing
```

154 tests across twelve areas:

- **Data contracts** — the target is binary, the zero-tenure fix holds, the train/validation/test
  splits are disjoint and stratified, and neither the target nor the customer ID can leak into features.
- **Validation** — missing columns, bad target values, malformed numerics, unknown categories and
  invalid `CostModel` values fail early with a readable message; out-of-range, NaN and infinite API
  inputs return a clean 422.
- **Integrity and provenance** — a tampered dataset raises `DatasetIntegrityError`, and the model
  report contains the git SHA, dataset hash, hyperparameters and library versions.
- **Feature engineering** — zero tenure never divides by zero, add-on counting is correct, and
  linked fields are permuted together so importance is not inflated by impossible rows.
- **Cost model** — the confusion-matrix arithmetic, that the tuned threshold never loses to 0.5 on
  validation, that an optimum between old grid points is found, and that test labels never enter the search, plus ECE and reliability-table checks.
- **Uncertainty** — the bootstrap is seeded and reproducible, its interval contains the point
  estimate, and the sensitivity grid's $50 / 30% cell equals the headline profit.
- **Operations** — liveness stays up and readiness fails when the model is missing, error bodies are
  structured, and a scoring failure leaks nothing into the response or the logs.
- **Survival** — the analysis runs end to end on the cleaned data, validation concordance is stored and
  above 0.5, Kaplan-Meier curves are valid and correctly ordered, hazard ratios point the expected
  way, the test split is never used, and the proportional-hazards check covers every covariate.
- **Drift** — a clean batch is `ok`, a shifted one alerts, unseen categories and missing values are
  counted, garbled input does not crash, `/drift` works without Evidently, and the module never calls
  training.
- **Explanations** — SHAP drivers map to original fields, grouped values are additive, a new
  month-to-month fiber customer is explained by `tenure` or `Contract`, and scoring survives a missing
  or failing explainer.
- **Edge cases** — single-class metrics fail loudly, all-identical predictions, threshold 0 / 1 and
  ties, and exact hand-computed values for every engineered feature.
- **API** — runs against a small deterministic fixture model built in `tests/conftest.py`, so it
  **never needs a trained artifact and is never skipped**. Covers schema validation, batch order,
  empty / oversized (1,001) / malformed requests, a behavioural check that a new month-to-month fiber
  customer outranks a two-year contract holder, and a missing or corrupt model file (the API starts
  degraded: `/health` says so and `/predict` returns 503).
- **Integration** — `tests/test_integration_training.py` runs the full training twice on the real
  dataset and asserts the same selected model, threshold and metrics both times.

CI runs lint, a full training run, and the suite with coverage on Python 3.12 and 3.11 on every push.
Coverage is 80.5% (859 of 1,067 statements; measured locally on Python 3.11 with the integration test, 2026-10-02)
and CI fails below 79% — the floor sits just under the current level, it is not a target. Most of the
uncovered code is figure rendering and the Evidently report, which the tests mostly skip.
