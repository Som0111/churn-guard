# Glossary

> Part of the [ChurnGuard README](../README.md). This page holds the detail; every number here is the same as in the README.

| Term | Meaning here |
|---|---|
| **ROC-AUC / PR-AUC** | How well the model ranks churners above stayers (1 is perfect, 0.5 is a coin flip for ROC-AUC) |
| **Brier score / ECE** | How close predicted probabilities are to observed rates (lower is better) |
| **Blanket campaign** | Send the offer to every customer; the no-model baseline |
| **Threshold** | Predicted-probability cut-off above which a customer gets an offer; chosen to maximise campaign profit |
| **SHAP value** | A field's contribution to one customer's score, in log-odds, relative to an average customer |
| **Hazard ratio** | Relative rate of leaving at any given tenure among customers still active (2 = twice the rate) |
| **Censored** | A customer who has not churned yet: we know they lasted at least their tenure, no more |
