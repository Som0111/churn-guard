"""API contract tests.

These run against the real fitted pipeline, so they double as an end-to-end
check that a raw customer dict survives feature engineering and scoring.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churnguard import config
from churnguard.api import app

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="model artifact missing - run `python -m churnguard.train`",
)

HIGH_RISK = {
    "tenure": 1,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "TechSupport": "No",
    "OnlineSecurity": "No",
    "MonthlyCharges": 95.0,
    "TotalCharges": 95.0,
}

LOW_RISK = {
    "tenure": 68,
    "Contract": "Two year",
    "InternetService": "DSL",
    "PaymentMethod": "Credit card (automatic)",
    "TechSupport": "Yes",
    "OnlineSecurity": "Yes",
    "MonthlyCharges": 60.0,
    "TotalCharges": 4080.0,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert 0 < body["threshold"] < 1


def test_root_redirects_to_the_docs(client):
    """Anyone opening the bare domain should land somewhere useful, not a 404."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (307, 308)
    assert response.headers["location"] == "/docs"


def test_predict_returns_a_valid_probability(client):
    response = client.post("/predict", json=HIGH_RISK)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}
    assert body["recommended_action"]


def test_model_separates_an_obvious_high_and_low_risk_customer(client):
    """A new month-to-month fiber customer must outrank a 5-year contract holder."""
    high = client.post("/predict", json=HIGH_RISK).json()["churn_probability"]
    low = client.post("/predict", json=LOW_RISK).json()["churn_probability"]
    assert high > low


def test_decision_follows_the_tuned_threshold(client):
    body = client.post("/predict", json=HIGH_RISK).json()
    assert body["will_churn"] == (body["churn_probability"] >= body["threshold_used"])


def test_batch_scoring_preserves_order(client):
    response = client.post(
        "/predict/batch", json={"customers": [HIGH_RISK, LOW_RISK]}
    )
    assert response.status_code == 200

    results = response.json()
    assert len(results) == 2
    assert results[0]["churn_probability"] > results[1]["churn_probability"]


def test_invalid_input_is_rejected(client):
    response = client.post("/predict", json={**HIGH_RISK, "Contract": "Lifetime"})
    assert response.status_code == 422


def test_negative_tenure_is_rejected(client):
    response = client.post("/predict", json={**HIGH_RISK, "tenure": -5})
    assert response.status_code == 422


def test_metrics_endpoint_exposes_the_training_report(client):
    body = client.get("/metrics").json()
    assert "business_impact" in body
    assert body["test_metrics_tuned_threshold"]["roc_auc"] > 0.75


def test_metrics_report_stores_ece_and_calibration_choice(client):
    body = client.get("/metrics").json()
    assert 0 <= body["validation_metrics_tuned_threshold"]["ece"] <= 1
    assert 0 <= body["test_metrics_tuned_threshold"]["ece"] <= 1
    assert body["calibration_method"]
