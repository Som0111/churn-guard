# Decisions

One short entry per fix: what was decided, why, and what it costs.

## 1. Train / validation / test, threshold tuned on validation
- **What:** 60/20/20 stratified split; CV on train picks the model, validation picks the profit threshold, test is
  scored once. The threshold search tries every unique predicted probability.
- **Why:** tuning the threshold on test made the headline profit optimistic while the README claimed test was untouched.
- **Trade-off:** less training data (60% instead of 80%) and a lower, more honest headline (+$16,250 → +$15,900).

## 2. Drop class weights instead of calibrating them
- **What:** compared balanced weights, no weights, and balanced + sigmoid / isotonic on validation; the unweighted
  model won on Brier + ECE and is the simplest.
- **Why:** the cost model multiplies probabilities by dollars, and balanced weights inflated the mean prediction from
  26.5% to 40.7%.
- **Trade-off:** the validation set now picks both the variant and the threshold (small selection effect); a fourth split
  would remove it at the price of less data.

## 3. `constraints.txt` over a lock file; one production Python
- **What:** `pyproject.toml` declares dependencies, `constraints.txt` pins them, Python 3.12 is production and 3.11 a CI
  compatibility check. The dataset is checked against a SHA-256.
- **Why:** open-ended minimums and four Python versions meant a fresh clone could not be trusted to reproduce results.
- **Trade-off:** the pins were resolved for 3.12/Linux but only run on 3.11/Windows locally, and some packages are capped
  (NumPy < 2.5, SciPy < 1.18) so one file serves both versions.

## 4. Validate early, and do not echo input
- **What:** schema checks on raw data, `CostModel` validation, bounded API inputs, and a custom 422 body.
- **Why:** bad data should fail with a readable message; a NaN request was crashing the API with a 500.
- **Trade-off:** only one cross-field rule (tenure 0 with a large `TotalCharges`), because the real data has no safe tighter bound.

## 5. Fixture model for API tests
- **What:** a small seeded model built in `conftest.py`; the full training run is a separate integration test.
- **Why:** API tests silently skipped when the real artifact was missing.
- **Trade-off:** the API tests check behaviour, not the real model's quality; the ~1 minute integration test slows the full suite.

## 6. Bootstrap the test set; sensitivity at a fixed threshold
- **What:** 1,000 seeded resamples for 90% intervals, and a success-rate × offer-cost grid.
- **Why:** one profit number hides both sampling noise and how much rests on assumed costs.
- **Trade-off:** the interval covers test-sample noise only, and the grid holds the threshold fixed, so it overstates
  losses for a team that would re-tune.

## 7. SHAP grouped back to original fields; `shap` optional
- **What:** contributions of engineered and one-hot features are folded onto the raw customer field; multi-source
  features are split equally. `shap` lives in `.[explain]`.
- **Why:** an analyst acts on "tenure" and "Contract", not on `is_new_customer`; the extra costs ~150 MB.
- **Trade-off:** the equal split is a convention. The extra must be installed deliberately (the live demo does install it and returns drivers).
- **Done when** ("a curl to the live API returns reasons"): met on https://churn-guard-api.onrender.com, checked 2026-10-02.

## 8. Drift verdict with SciPy, Evidently only for the HTML report
- **What:** KS and chi-square tests decide `ok / warning / alert`; Evidently (`.[monitor]`) renders the report.
- **Why:** Evidently adds ~430 MB and would not fit the free Render build; the verdict is two standard tests.
- **Trade-off:** two code paths can disagree on a borderline column, and the thresholds are untuned to real traffic.

## 9. Survival analysis as a report, not an endpoint
- **What:** Kaplan-Meier and a Cox model in an optional `.[analysis]` extra; `TotalCharges` excluded from the Cox model.
- **Why:** it answers "when", not "whether", and nothing in the serving path needs it.
- **Trade-off:** proportional hazards partly fails, so some hazard ratios are averages; the Kaplan-Meier curves carry the conclusions.

## 10. Separate liveness and readiness; log class names, not messages
- **What:** `/health/live`, `/health/ready`, `/model-info`, compact `/metrics`, a non-root image pinned by digest, a CI smoke test.
- **Why:** a live process with no model is not ready; sklearn and pandas error messages can quote customer values.
- **Trade-off:** server-side debugging is harder without messages. CI builds and smoke-tests the container, and also checks that it knows its commit (a build with no `.git` needs `GIT_SHA` or `RENDER_GIT_COMMIT`).

## 11. Say what the evidence supports, and shuffle linked fields together
- **What:** removed causal wording, added "what this project does not prove", and permuted `tenure` and `TotalCharges` as one block.
- **Why:** a per-column shuffle of `tenure` produced an AUC drop (0.375) larger than the model's headroom because it created
  impossible rows; "highest-leverage intervention" was not supported by observational data.
- **Trade-off:** the importance table now merges two fields into one row, and the README is longer and more hedged.
