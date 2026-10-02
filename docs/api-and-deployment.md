# API, Docker and deployment

> Part of the [ChurnGuard README](../README.md). This page holds the detail; every number here is the same as in the README.

## Full quickstart

```bash
git clone https://github.com/som0111/churn-guard.git
cd churn-guard

pip install -c constraints.txt -e ".[dev,explain]"   # pinned install (+SHAP drivers)
python -m churnguard.train  # download data, train, evaluate, write figures (~35s)
pytest                      # 154 tests (~3 min; add -m "not integration" to skip the training run)
uvicorn churnguard.api:app --reload
```

The `explain` extra (SHAP) adds about 150 MB; for a lean install use `pip install -c constraints.txt -e ".[dev]"` and the API
scores without `top_drivers`. Other optional extras: `monitor` (Evidently HTML drift report) and `analysis` (survival).

**Production Python is 3.12** (Docker, Render and the instructions here). CI runs 3.12 and also 3.11
as a compatibility check. `pyproject.toml` is the single source of truth for dependencies;
[`constraints.txt`](../constraints.txt) pins every resolved version, so a fresh clone installs exactly
what the reported metrics were produced with. To regenerate the pins after changing dependencies:
resolve `pip install ".[dev,explain]"` for Python 3.12 and write the result back to `constraints.txt`
(NumPy is held below 2.5 and SciPy below 1.18 so the same pins also install on 3.11).

Interactive API docs: <http://127.0.0.1:8000/docs>

## Scoring a customer

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 1, "Contract": "Month-to-month", "InternetService": "Fiber optic",
       "PaymentMethod": "Electronic check", "TechSupport": "No", "OnlineSecurity": "No",
       "MonthlyCharges": 95.0, "TotalCharges": 95.0}'
```

```json
{
  "churn_probability": 0.7092,
  "will_churn": true,
  "risk_band": "high",
  "threshold_used": 0.3837,
  "recommended_action": "Priority outreach: call within 48h and offer a contract upgrade.",
  "top_drivers": [
    {"feature": "tenure",          "direction": "raises", "magnitude": 1.7518},
    {"feature": "Contract",        "direction": "raises", "magnitude": 0.6835},
    {"feature": "InternetService", "direction": "raises", "magnitude": 0.6016}
  ]
}
```

The API returns an **action** — and, when the SHAP extra is installed, the fields behind the score — not just a
number, so a retention analyst can act on it without knowing what a probability is.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | Redirects to the interactive docs |
| `POST /predict` | Score one customer |
| `POST /predict/batch` | Score up to 1,000 in one call |
| `GET /health/live` | Process is up (says nothing about the model) |
| `GET /health/ready` | 200 only when a model is loaded, otherwise 503 |
| `GET /health` | Combined status; kept for Render's health check |
| `GET /model-info` | Model version, training time, threshold, dataset SHA-256, calibration method |
| `GET /metrics` | Compact summary of held-out results (the full report stays in `reports/metrics.json`) |
| `POST /drift` | Compare a recent batch of raw records to the training reference |

## Docker

```bash
docker build -t churnguard .
docker run -p 8000:8000 churnguard
```

Errors share one shape, `{"detail": ..., "error_code": ...}`. A scoring failure returns a generic 500 and
the server log records only the exception class and batch size — never the request or the exception
message, which can quote customer values.

The image is based on `python:3.12.7-slim-bookworm` (pinned by digest), runs as a non-root user,
has a `HEALTHCHECK` on `/health/ready`, installs with the pinned `constraints.txt` (including
the `explain` extra; build with `--build-arg EXTRAS=` for a slim image without SHAP drivers), and
trains the model at build time, so the container starts ready to serve. Pass
`--build-arg GIT_SHA=$(git rev-parse --short HEAD)` to record the commit in the model version (without it the
version reads `unknown-<time>`; the build context has no `.git`). CI builds
this image with the commit SHA, starts it, runs [`scripts/smoke_test.sh`](../scripts/smoke_test.sh) against
`/health/ready` and `/predict`, and checks that `/model-info` reports that commit.

## Deployment

The live instance runs on Render's free tier, configured by [`render.yaml`](../render.yaml):

```yaml
buildCommand: pip install -c constraints.txt -e ".[explain]" && python -m churnguard.train --skip-figures
startCommand: uvicorn churnguard.api:app --host 0.0.0.0 --port $PORT
```

**The model is trained during the build rather than committed.** Two reasons: the pickled artifact
is always produced by the exact scikit-learn build that will load it, sidestepping version-skew
bugs — and every deploy re-proves that the pipeline runs end to end on a clean machine.

**Checked against the live instance (https://churn-guard-api.onrender.com) on 2026-10-02:** `/health/live`,
`/health/ready`, `/model-info`, `/metrics` and `/predict` all respond, and `/predict` returns `top_drivers`. The sample
customer above scores `0.7092` there — the same as a local run, so the fixed seed and pinned libraries reproduce the
probability across Windows/3.11 and Linux/Render. The threshold agrees only to about 10 decimal places
(0.38369581667 live vs 0.38369581670 local): same metrics, not bit-identical floats. `/model-info` shows which commit
the running model was built from (`model_version` is `<short commit>-<UTC time>`); the commit comes from `GIT_SHA`
or Render's `RENDER_GIT_COMMIT` because neither a Docker build nor a Render build has a `.git` directory.

Pushing to `main` triggers CI and a redeploy. Free instances sleep after 15 minutes idle, so the
first request back takes ~50 seconds.
