"""Health probes, model info, and safe structured errors."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient
from test_api import HIGH_RISK

from churnguard import api, config


def test_live_is_always_up(client):
    assert client.get("/health/live").json() == {"status": "alive"}


def test_ready_reports_the_loaded_model(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready" and response.json()["model_version"]


def test_legacy_health_still_works_for_render(client):
    assert client.get("/health").json()["status"] == "ok"


def test_metrics_is_much_smaller_than_the_offline_report(client):
    assert len(client.get("/metrics").content) < 1500


# --------------------------------------------------------------------------- #
# No model: live stays up, ready fails
# --------------------------------------------------------------------------- #
@pytest.fixture
def no_model_client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(api, "MODEL", {})
    with TestClient(api.app) as client:
        yield client


def test_readiness_fails_when_the_model_is_missing(no_model_client):
    response = no_model_client.get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "model_loaded": False}


def test_liveness_survives_a_missing_model(no_model_client):
    assert no_model_client.get("/health/live").status_code == 200


def test_model_info_and_scoring_return_structured_503s(no_model_client):
    for response in (
        no_model_client.get("/model-info"),
        no_model_client.post("/predict", json=HIGH_RISK),
    ):
        assert response.status_code == 503
        assert response.json()["error_code"] == "model_unavailable" and response.json()["detail"]


# --------------------------------------------------------------------------- #
# Structured, safe errors
# --------------------------------------------------------------------------- #
def test_validation_errors_are_structured(client):
    body = client.post("/predict", json={**HIGH_RISK, "tenure": -1}).json()
    assert body["error_code"] == "validation_error" and isinstance(body["detail"], list)


def test_unknown_route_is_structured(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found", "error_code": "not_found"}


class _Exploding:
    """A model whose failure message quotes customer data, like a real pandas/sklearn error."""

    def predict_proba(self, frame):
        raise ValueError("could not convert string to float: 'Lawrence-4821-secret'")


def test_inference_failure_returns_a_safe_500_and_logs_no_payload(client, monkeypatch, caplog):
    monkeypatch.setitem(api.MODEL, "pipeline", _Exploding())
    with caplog.at_level(logging.DEBUG):
        response = client.post("/predict", json={**HIGH_RISK, "tenure": 7})

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Scoring failed. The error has been logged.",
        "error_code": "internal_error",
    }
    assert "secret" not in response.text
    # server-side: the failure is recorded by class and size only
    assert "Inference failed (ValueError) on a batch of 1" in caplog.text
    assert "Lawrence" not in caplog.text and "secret" not in caplog.text
    assert "95.0" not in caplog.text  # no field values from the request


def test_unexpected_errors_become_a_generic_500(client, monkeypatch, caplog):
    def boom(_request):
        raise RuntimeError("password=hunter2")

    monkeypatch.setattr(api.drift, "analyse", lambda *a, **k: boom(None))
    records = [{"tenure": 1}] * config.DRIFT_MIN_ROWS
    quiet = TestClient(api.app, raise_server_exceptions=False)
    with caplog.at_level(logging.DEBUG):
        response = quiet.post("/drift", json={"records": records})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error.", "error_code": "internal_error"}
    assert "hunter2" not in caplog.text and "hunter2" not in response.text
