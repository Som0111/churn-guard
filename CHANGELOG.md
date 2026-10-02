# Changelog

Eleven fixes, in the order they were made. Each lists what changed and the real numbers before and after.
"Test" means the held-out test split (1,409 customers). Dollar figures are projections under assumed costs
($50 offer, $500 customer value, 30% save rate), not observed outcomes.

## Fix 1 — Clean evaluation protocol

- Split into 60/20/20 train / validation / test. Model selection on train (CV), threshold tuned on
  **validation**, test scored once at the frozen threshold. Threshold search runs over every unique predicted
  probability (plus 0 and 1) instead of a 200-point grid.
- The old README said the test set was "never touched" while the threshold was tuned on it.

| | Before | After |
|---|---|---|
| Test profit at tuned threshold | +$16,250 | +$15,900 |
| Tuned threshold | 0.5419 | 0.6429 |
| Test ROC-AUC | 0.8458 | 0.8464 |
| Precision / recall | 53.4% / 77.0% | 58.0% / 66.6% |
| Tests | 22 | 26 |

## Fix 2 — Calibration audit

- `class_weight="balanced"` predicted a 40.7% mean churn rate for a 26.5% population. Compared four variants on
  validation; the unweighted model won (sigmoid tied on Brier but had higher ECE).
- Added ECE and a reliability table; wrote `reports/calibration_comparison.json`.

| | Before | After |
|---|---|---|
| Validation ECE | 0.142 | 0.019 |
| Validation Brier | 0.1655 | 0.1369 |
| Test Brier | 0.1648 | 0.1359 |
| Test ECE | not measured | 0.020 |
| Tuned threshold | 0.6429 | 0.3837 |
| Test profit at tuned threshold | +$15,900 | +$15,900 |
| Test profit at 0.50 | +$15,400 | +$15,050 |
| Precision / recall | 58.0% / 66.6% | 57.4% / 67.7% |
| Tests | 26 | 29 |

## Fix 3 — Reproducible environment and provenance

- Python 3.12 is the single production version (Docker, Render, README); CI runs 3.12 plus 3.11 for
  compatibility. `pyproject.toml` is the only dependency source; `constraints.txt` pins every resolved version.
- Dataset SHA-256 verified on download and every load (`DatasetIntegrityError` on mismatch).
- Model report records git SHA, timestamp, `model_version`, dataset hash, hyperparameters, calibration and
  threshold method, library versions and split counts.
- Metrics unchanged. The NumPy/joblib deprecation warning disappeared with the pins. Tests 29 → 33.

## Fix 4 — Input and config validation

- Raw-data schema check, `CostModel` validation, bounded API inputs, one conservative consistency check.
- Found and fixed a bug: NaN or Infinity in a request crashed the API with a **500** (FastAPI echoed the
  non-JSON input in its 422 body). Now a clean **422** that does not echo the payload.
- Tests 33 → 63.

## Fix 5 — Test suite hardening

- API tests run against a deterministic fixture model and are never skipped (verified with the real model
  file removed). Added edge-case tests, exact engineered-feature values, a train-twice reproducibility test, and a
  coverage floor.
- A corrupt model file used to crash startup; the API now starts degraded (503 on `/predict`).
- Tests 63 → 83. Coverage 78%, CI floor 77%.

## Fix 6 — Profit uncertainty and cost sensitivity

- Seeded bootstrap (1,000 resamples) and a success-rate × offer-cost grid.

| | Point | 90% interval |
|---|---|---|
| Model campaign profit | +$15,900 | $13,000 to $18,600 |
| Blanket campaign profit | −$14,350 | −$18,850 to −$10,450 |
| ROC-AUC | 0.846 | 0.828 to 0.864 |

- The campaign loses money in 45 of 144 grid cells; the model beats blanket mailing in 140 of 144.
- Tests 83 → 87.

## Fix 7 — SHAP top drivers in the API

- `/predict` returns `top_drivers` (three fields, direction, magnitude), with contributions of engineered
  features folded back onto the original customer field. `shap` is an optional extra.
- `/predict` p50 latency 19.7 ms → 35.0 ms (one machine, in-process). Install size +154 MB.
- Tests 87 → 102.

## Fix 8 — Drift monitoring

- Reference sample saved at training time; `POST /drift`; `python -m churnguard.drift --demo`. The verdict uses
  SciPy tests, so it works without Evidently (which adds ~430 MB and only renders the HTML report).
- Demo: clean batch `ok` (0 of 19 fields drifted); simulated shift `alert` (6 of 19, a narrow margin over the
  30% line). Tests 102 → 120.

## Fix 9 — Survival analysis

- Kaplan-Meier by contract and internet service; Cox model with validation concordance **0.855**.
  Proportional hazards fails for 2 of 13 covariates (`Contract_One year`, `n_addon_services`).
  Optional `.[analysis]` extra; report only. Tests 120 → 129.

## Fix 10 — API and container hygiene

- `/health/live`, `/health/ready`, `/model-info`; `/metrics` is a compact summary (**17,402 → 527 bytes**);
  structured error bodies; inference errors log the exception class and batch size only.
- Docker: non-root user, `HEALTHCHECK`, base image pinned by digest; CI builds the image and smoke-tests it.
- CI's container job (build, start, smoke test, non-root check) passes.
- Tests 129 → 139.

## Fix 11 — README honesty, docs and resume pack

- Removed causal language; added "What this project does not prove"; described `spend_vs_current_ratio` as
  current charge over lifetime average, not a price rise.
- **Permutation-importance anomaly:** shuffling `tenure` alone reported an AUC drop of **0.375**, more than the
  model's headroom above chance (0.346). It manufactured impossible rows (`avg_monthly_spend` 99th percentile
  $115 → $6,327; AUC after the shuffle 0.452). `tenure` and `TotalCharges` are now shuffled as one block.

| | Before | After |
|---|---|---|
| Top importance | `tenure` alone: 0.375 | `tenure` + `TotalCharges` jointly: 0.110 |
| `Contract` | 0.046 | 0.051 |
| `MonthlyCharges` | 0.032 | 0.033 |
| `InternetService` | 0.024 | 0.023 |

- Live Render instance (https://churn-guard-api.onrender.com) checked on 2026-10-02: scores `0.7092` (same as local)
  and returns `top_drivers`. Fix 7's "Done when" (a curl to the live API returns reasons, not only a score) is **met**.
- Tests 139 → 142.

## Fix 11b — Commit SHA in builds without `.git`

- A Docker or Render build has no `.git`, so `model_version` read `unknown-<time>`. The commit now comes from
  `GIT_SHA` (Docker build arg, passed by CI as the first 7 characters of `github.sha`) or `RENDER_GIT_COMMIT`; an empty
  value or the literal `unknown` counts as not provided. `git_dirty` is `null` (not `false`) when it cannot be known.
- CI now fails if the built image's `/model-info` reports `unknown` or a different commit.
- Tests 142 → 150 (commit lookup: env var, Render fallback, empty/`unknown` ignored, git fallback, model version).
- Coverage re-measured at **80.5%** (859 of 1,067 statements; it had gone stale at 78%). CI floor raised 77% → 79%.
