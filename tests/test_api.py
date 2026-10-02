"""API contract tests.

They run against a small deterministic fixture model (see ``conftest.py``), so
they never depend on a trained artifact and are never skipped. A raw customer
dict still goes through the real feature engineering and scoring path.
"""

from __future__ import annotations

import json

import pytest

from churnguard import config

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


def test_metrics_endpoint_returns_a_compact_summary(client):
    body = client.get("/metrics").json()
    assert body["test"]["roc_auc"] > 0.75
    assert 0 <= body["test"]["ece"] <= 1
    assert body["calibration_method"]
    assert "model" in body["test_campaign_profit"]
    # the whole offline report (leaderboard, per-feature importances, ...) is not served
    assert not {"leaderboard", "top_drivers", "provenance", "business_impact"} & body.keys()


def test_model_info_names_the_model_and_its_data(client):
    body = client.get("/model-info").json()
    assert body["dataset_sha256"] == config.DATA_SHA256
    assert body["model_version"] and body["trained_at"] and body["calibration_method"]
    assert 0 < body["threshold"] < 1


@pytest.mark.parametrize(
    "field, value",
    [
        ("tenure", 101),
        ("MonthlyCharges", -1.0),
        ("MonthlyCharges", 1000.01),
        ("TotalCharges", -1.0),
        ("TotalCharges", 1e9),
    ],
)
def test_out_of_range_numbers_are_rejected(client, field, value):
    response = client.post("/predict", json={**HIGH_RISK, field: value})
    assert response.status_code == 422
    assert field in str(response.json()["detail"])


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("field", ["MonthlyCharges", "TotalCharges"])
def test_nan_and_infinity_are_rejected(client, field, literal):
    body = "{" + ", ".join(
        f'"{k}": {literal if k == field else json.dumps(v)}' for k, v in HIGH_RISK.items()
    ) + "}"
    response = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_zero_tenure_with_large_total_is_rejected_with_a_clear_message(client):
    response = client.post(
        "/predict", json={**HIGH_RISK, "tenure": 0, "MonthlyCharges": 50.0, "TotalCharges": 2000.0}
    )
    assert response.status_code == 422
    assert "tenure 0" in response.json()["detail"][0]["msg"]


def test_zero_tenure_edge_cases_are_still_accepted(client):
    for total in (0.0, 30.0, 50.0):  # none, partial first bill, exactly one month
        r = client.post("/predict", json={**HIGH_RISK, "tenure": 0, "MonthlyCharges": 50.0, "TotalCharges": total})
        assert r.status_code == 200, total


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_oversized_batch_is_rejected(client):
    too_many = {"customers": [HIGH_RISK] * 1001}
    assert client.post("/predict/batch", json=too_many).status_code == 422


def test_batch_at_the_limit_is_accepted(client):
    response = client.post("/predict/batch", json={"customers": [HIGH_RISK] * 1000})
    assert response.status_code == 200 and len(response.json()) == 1000


def test_malformed_json_is_rejected(client):
    response = client.post(
        "/predict", content="{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


def test_missing_required_field_names_the_field(client):
    body = {k: v for k, v in HIGH_RISK.items() if k != "MonthlyCharges"}
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    assert "MonthlyCharges" in str(response.json()["detail"])


def test_wrong_type_is_rejected(client):
    assert client.post("/predict", json={**HIGH_RISK, "tenure": "soon"}).status_code == 422


def test_unknown_batch_member_fails_the_whole_request(client):
    response = client.post(
        "/predict/batch", json={"customers": [HIGH_RISK, {**LOW_RISK, "Contract": "Weekly"}]}
    )
    assert response.status_code == 422
