"""Drift monitoring: quiet on unshifted data, loud on shifted data, never retrains."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import joblib
import numpy as np
import pytest
from conftest import synthetic_customers

from churnguard import api, config, drift
from churnguard.drift import analyse, detect, score_frame, simulate_shift


@pytest.fixture(scope="module")
def world(fixture_artifacts):
    """The fixture model's artifact plus unseen customers from the same process."""
    artifact = joblib.load(fixture_artifacts[0])
    X, _ = synthetic_customers(n=3000, seed=11)  # a fresh seed: never in the reference
    return artifact, X


def test_unshifted_sample_is_ok(world):
    artifact, X = world
    result = analyse(artifact, X.iloc[:600])
    assert result["status"] == "ok", result["drifted_features"]
    assert result["share_drifted"] < config.DRIFT_WARNING_SHARE
    assert not result["prediction_drift"]["drifted"]


def test_shifted_sample_raises_an_alert(world):
    artifact, X = world
    result = analyse(artifact, simulate_shift(X.iloc[:600]))
    assert result["status"] == "alert"
    assert result["share_drifted"] >= config.DRIFT_ALERT_SHARE
    drifted = {d["feature"] for d in result["drifted_features"]}
    assert {"tenure", "MonthlyCharges", "InternetService", "Contract"} <= drifted
    assert result["prediction_drift"]["drifted"]


def test_an_unseen_category_and_missing_values_are_reported(world):
    artifact, X = world
    quality = analyse(artifact, simulate_shift(X.iloc[:600]))["data_quality"]
    assert 0.10 < quality["unknown_category_rate"]["PaymentMethod"] < 0.20  # ~15% "Crypto"
    assert quality["overall_unknown_category_rate"] > 0
    assert 0.02 < quality["missing_rate"]["MonthlyCharges"] < 0.09  # ~5% blanked


def test_unseen_category_counts_as_feature_drift(world):
    artifact, X = world
    result = analyse(artifact, simulate_shift(X.iloc[:600]))
    assert any(d["feature"] == "PaymentMethod" for d in result["drifted_features"])


def test_a_clean_batch_reports_no_data_quality_issues(world):
    artifact, X = world
    quality = analyse(artifact, X.iloc[:600])["data_quality"]
    assert quality["overall_missing_rate"] == 0
    assert quality["overall_unknown_category_rate"] == 0


def test_a_vanished_numeric_field_counts_as_drift(world):
    artifact, X = world
    broken = X.iloc[:300].copy()
    broken["tenure"] = np.nan
    drifted = {d["feature"] for d in analyse(artifact, broken)["drifted_features"]}
    assert "tenure" in drifted


def test_garbled_numbers_and_missing_columns_do_not_crash(world):
    artifact, X = world
    messy = X.iloc[:100].drop(columns=["Contract"]).copy()
    messy["MonthlyCharges"] = messy["MonthlyCharges"].astype(object)
    messy.loc[messy.index[:10], "MonthlyCharges"] = "n/a"
    quality = analyse(artifact, messy)["data_quality"]
    assert quality["missing_rate"]["Contract"] == 1.0
    assert quality["missing_rate"]["MonthlyCharges"] >= 0.1


def test_too_few_rows_is_refused(world):
    artifact, X = world
    with pytest.raises(ValueError, match="at least"):
        analyse(artifact, X.iloc[: config.DRIFT_MIN_ROWS - 1])


def test_every_field_and_the_prediction_get_a_p_value(world):
    artifact, X = world
    p_values = analyse(artifact, X.iloc[:300])["p_values"]
    assert set(p_values) == {*config.FEATURES, drift.PREDICTION}
    assert all(0 <= p <= 1 for p in p_values.values())


def test_status_thresholds():
    assert drift.status_for(0.0, False) == "ok"
    assert drift.status_for(0.0, True) == "warning"
    assert drift.status_for(config.DRIFT_WARNING_SHARE, False) == "warning"
    assert drift.status_for(config.DRIFT_ALERT_SHARE, False) == "alert"


def test_detect_is_deterministic(world):
    artifact, X = world
    frame = score_frame(artifact, simulate_shift(X.iloc[:300]))
    assert detect(artifact["drift_reference"], frame) == detect(artifact["drift_reference"], frame)


def test_drift_module_never_calls_training():
    """Detect and alert only: nothing in the module may import the trainer."""
    source = pathlib.Path(drift.__file__).read_text(encoding="utf-8")
    assert "churnguard.train" not in source and "import train" not in source


# --------------------------------------------------------------------------- #
# Through the API
# --------------------------------------------------------------------------- #
def test_drift_endpoint_ok_and_alert(client, world):
    _, X = world
    ok = client.post("/drift", json={"records": X.iloc[:600].to_dict("records")})
    assert ok.status_code == 200 and ok.json()["status"] == "ok"

    shifted = simulate_shift(X.iloc[:600]).astype(object).where(lambda d: d.notna(), None)
    alert = client.post("/drift", json={"records": shifted.to_dict("records")})
    assert alert.status_code == 200
    body = alert.json()
    assert body["status"] == "alert" and body["drifted_features"]
    assert body["data_quality"]["unknown_category_rate"]["PaymentMethod"] > 0


def test_drift_endpoint_rejects_tiny_batches(client, world):
    _, X = world
    response = client.post("/drift", json={"records": X.iloc[:10].to_dict("records")})
    assert response.status_code == 422


def test_drift_endpoint_needs_no_evidently(client, world, monkeypatch):
    """Evidently only renders the HTML report; the endpoint must work without it."""
    monkeypatch.setitem(sys.modules, "evidently", None)  # makes `import evidently` fail
    _, X = world
    response = client.post("/drift", json={"records": X.iloc[:100].to_dict("records")})
    assert response.status_code == 200


def test_html_report_without_evidently_explains_how_to_install(world, monkeypatch):
    monkeypatch.setitem(sys.modules, "evidently", None)
    artifact, X = world
    with pytest.raises(RuntimeError, match=r"\.\[monitor\]"):
        drift.write_html_report(artifact["drift_reference"], score_frame(artifact, X.iloc[:100]))


def test_drift_returns_503_without_a_reference(client, monkeypatch):
    monkeypatch.setitem(api.MODEL, "drift_reference", None)
    X, _ = synthetic_customers(n=100, seed=1)
    assert client.post("/drift", json={"records": X.to_dict("records")}).status_code == 503


@pytest.mark.skipif(importlib.util.find_spec("evidently") is None, reason="evidently not installed")
def test_html_report_is_written_when_evidently_is_installed(world, tmp_path):
    artifact, X = world
    path = drift.write_html_report(
        artifact["drift_reference"],
        score_frame(artifact, simulate_shift(X.iloc[:300])),
        tmp_path / "r.html",
    )
    assert path.exists() and path.stat().st_size > 10_000
