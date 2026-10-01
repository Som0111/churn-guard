"""A missing or corrupt model file must degrade the API, not crash it."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_api import HIGH_RISK

from churnguard import api, config


@pytest.fixture(params=["missing", "corrupt"])
def broken_client(request, tmp_path, monkeypatch):
    path = tmp_path / "churn_pipeline.joblib"
    if request.param == "corrupt":
        path.write_bytes(b"this is not a pickle")
    monkeypatch.setattr(config, "MODEL_PATH", path)
    monkeypatch.setattr(api, "MODEL", {})  # isolate from any other client's loaded model
    with TestClient(api.app) as client:
        yield client


def test_health_reports_degraded(broken_client):
    body = broken_client.get("/health").json()
    assert body["status"] == "degraded" and body["model_loaded"] is False


def test_predict_returns_503_with_guidance(broken_client):
    response = broken_client.post("/predict", json=HIGH_RISK)
    assert response.status_code == 503
    assert "train" in response.json()["detail"]


def test_batch_returns_503(broken_client):
    response = broken_client.post("/predict/batch", json={"customers": [HIGH_RISK]})
    assert response.status_code == 503


def test_load_model_raises_a_clear_error_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "nope.joblib")
    with pytest.raises(FileNotFoundError, match="churnguard.train"):
        api.load_model()
