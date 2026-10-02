# Calibration

> Part of the [ChurnGuard README](../README.md). This page holds the detail; every number here is the same as in the README.

![Calibration](../reports/figures/calibration.png)

The cost model multiplies predicted probabilities by dollar values, so miscalibrated probabilities
would silently mis-price the campaign. This was audited on the validation split
([`reports/calibration_comparison.json`](../reports/calibration_comparison.json)), comparing four ways of
producing probabilities from the selected logistic regression. Calibrators are fit by 5-fold CV on the
training split only.

| Variant | Brier | ECE | Mean predicted (observed 0.265) |
|---|---|---|---|
| `class_weight="balanced"` (the original) | 0.1655 | **0.142** | 0.407 |
| **No class weights** ✅ | **0.1369** | **0.019** | 0.260 |
| Balanced + sigmoid | 0.1368 | 0.022 | 0.261 |
| Balanced + isotonic | 0.1372 | 0.032 | 0.259 |

**The original `balanced` model was not calibrated:** it predicted a 40.7% average churn rate for a
population that churns at 26.5%, and its Brier score (0.1655) was only modestly better than the 0.195
a constant base-rate predictor gets. Winner: **no class weights** — lowest Brier + ECE (sigmoid ties on Brier,
0.1368 vs 0.1369, but has higher ECE and adds a calibrator for no gain). On the test
split the winner scores Brier 0.136 and ECE 0.020. Ranking quality is essentially unchanged (test ROC-AUC 0.8464 → 0.8463, PR-AUC 0.6563 → 0.6548); what moved is the probability scale, so the profit-tuned threshold fell from 0.64 to
0.38. ECE here uses 10 equal-count bins, and the validation split was used both to pick the variant and
to tune the threshold, so the figures carry a small selection effect.
