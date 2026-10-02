# Monitoring

> Part of the [ChurnGuard README](../README.md). This page holds the detail; every number here is the same as in the README.

![Drift demo](../reports/figures/drift_demo.png)

A deployed model degrades silently when the customers it sees stop looking like the ones it was
built on. `POST /drift` takes a batch of recent raw customer records (at least 50) and compares them to
a 1,000-row reference sample of the **validation split**, saved inside the model artifact at train time
together with the model's own predictions on it.

- **Feature drift.** Kolmogorov–Smirnov test for numeric fields, chi-square for categorical ones
  (`SeniorCitizen` counts as categorical). A field has drifted when `p < 0.01`.
- **Prediction drift.** The same KS test on the predicted churn probability.
- **Status.** `ok` below 10% of fields drifted, `warning` from 10% (or if predictions drifted),
  `alert` from 30%. The thresholds live in [`config.py`](../src/churnguard/config.py).
- **Data quality.** The response also reports each field's missing rate and the share of values outside
  the known categories. The records are deliberately *not* validated like `/predict` inputs, because new
  categories and missing values are exactly what this endpoint exists to see.
- **Detect and alert only.** Nothing retrains or changes the model.

```bash
python -m churnguard.drift --demo      # clean vs. simulated-shift batch -> reports/drift_demo.json,
                                       # reports/figures/drift_demo.png, reports/drift_report.html
```

The demo takes 500 test customers, then makes a shifted copy: tenure and total charges scaled to
40%, monthly charges up 25%, more fiber and month-to-month customers, 15% paying with an unseen
`Crypto` method, and 5% of monthly charges blanked. Result: the clean batch is **ok** (0 of 19 fields
drifted); the shifted batch is **alert** (6 of 19 drifted: `tenure`, `MonthlyCharges`, `TotalCharges`,
`InternetService`, `Contract`, `PaymentMethod`; predicted risk up from 26.3% to 35.9%). That alert is
close to its line (6 of 19 is 31.6% against a 30% threshold), so treat the thresholds as a starting
point to tune on real traffic, not a calibrated alarm.

**Evidently is optional.** The drift verdict uses SciPy, so `/drift` works in the lean install
(including the live Render deployment). Evidently only renders the HTML report
(`pip install -c constraints.txt ".[monitor]"`); it adds about 430 MB when installed, which is why it is
kept out of the Docker and Render builds. The HTML file is not committed (4 MB); the chart above is drawn
from the demo's own results, not an Evidently screenshot. The Evidently report is configured with the
same tests and p-value, but it makes its own pass over the data, so its drifted-column list is the one
to cross-check against, not a second opinion on the status.
