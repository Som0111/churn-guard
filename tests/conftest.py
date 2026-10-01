"""Shared fixtures: a tiny deterministic model so API tests never need a trained artifact."""

from __future__ import annotations

from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from churnguard import api, config, evaluate
from churnguard.drift import build_reference
from churnguard.explain import make_background
from churnguard.train import build_pipeline, build_provenance


def synthetic_customers(n: int = 900, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Customers whose churn depends on contract, internet type and tenure."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {col: rng.choice(config.ALLOWED_CATEGORIES[col], n) for col in config.CATEGORICAL_FEATURES}
    )
    df["tenure"] = rng.integers(0, 73, n)
    df["MonthlyCharges"] = rng.uniform(20, 110, n).round(2)
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"] * rng.uniform(0.9, 1.1, n)).round(2)
    df["SeniorCitizen"] = rng.integers(0, 2, n)

    logit = (
        -1.0
        + 2.0 * (df["Contract"] == "Month-to-month")
        + 1.0 * (df["InternetService"] == "Fiber optic")
        - 0.05 * df["tenure"]
    )
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-logit))), name=config.TARGET)
    return df[config.FEATURES], y


@pytest.fixture(scope="session")
def fixture_artifacts(tmp_path_factory):
    """Fit the small model once; write the artifact and a matching report to a temp dir."""
    X, y = synthetic_customers()
    X_tr, X_ho, y_tr, y_ho = X.iloc[:650], X.iloc[650:], y.iloc[:650], y.iloc[650:]

    model = build_pipeline(LogisticRegression(max_iter=1000, random_state=0)).fit(X_tr, y_tr)
    proba = model.predict_proba(X_ho)[:, 1]
    impact = evaluate.optimize_threshold(y_ho.to_numpy(), proba)
    threshold = impact["optimal_threshold"]
    metrics = evaluate.classification_metrics(y_ho.to_numpy(), proba, threshold)
    provenance = build_provenance(
        model,
        "no_class_weight",
        {"train": y_tr, "validation": y_ho, "test": y_ho},
        datetime(2026, 1, 1, tzinfo=UTC),
    )

    out = tmp_path_factory.mktemp("fixture_model")
    artifact = out / "churn_pipeline.joblib"
    joblib.dump(
        {
            "model_version": provenance["model_version"],
            "pipeline": model,
            "threshold": threshold,
            "model_name": "logistic_regression",
            "calibration": "no_class_weight",
            "explainer_background": make_background(model, X_tr),
            "drift_reference": build_reference(model, X_ho),
        },
        artifact,
    )
    report = out / "metrics.json"
    import json

    report.write_text(
        json.dumps(
            {
                "provenance": provenance,
                "calibration_method": "no_class_weight",
                "validation_metrics_tuned_threshold": metrics,
                "test_metrics_tuned_threshold": metrics,
                "business_impact": impact,
            }
        ),
        encoding="utf-8",
    )
    return artifact, report


@pytest.fixture(scope="module")
def client(fixture_artifacts):
    artifact, report = fixture_artifacts
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "MODEL_PATH", artifact)
        mp.setattr(config, "METRICS_PATH", report)
        with TestClient(api.app) as test_client:
            yield test_client
