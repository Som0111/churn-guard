"""Full training end to end, twice: the pipeline must run on real data and be reproducible."""

from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest
from test_api import HIGH_RISK

from churnguard import config, explain, train
from churnguard.api import Customer

pytestmark = pytest.mark.integration


def _train_into(tmp_path, monkeypatch, name):
    out = tmp_path / name
    out.mkdir()
    monkeypatch.setattr(config, "MODEL_PATH", out / "model.joblib")
    monkeypatch.setattr(config, "METADATA_PATH", out / "model_card.json")
    monkeypatch.setattr(config, "METRICS_PATH", out / "metrics.json")
    monkeypatch.setattr(config, "CALIBRATION_PATH", out / "calibration.json")
    report = train.main(skip_figures=True)
    return report, out


def test_training_runs_and_is_reproducible(tmp_path, monkeypatch):
    first, out1 = _train_into(tmp_path, monkeypatch, "a")
    second, _ = _train_into(tmp_path, monkeypatch, "b")

    assert (out1 / "model.joblib").exists()
    assert json.loads((out1 / "metrics.json").read_text())["provenance"]["model_version"]

    for key in (
        "selected_model",
        "calibration_method",
        "test_metrics_tuned_threshold",
        "validation_metrics_tuned_threshold",
        "business_impact",
    ):
        assert first[key] == second[key], key
    assert first["test_metrics_tuned_threshold"]["roc_auc"] > 0.80


def test_trained_model_explains_a_high_risk_customer(tmp_path, monkeypatch):
    _, out = _train_into(tmp_path, monkeypatch, "c")
    artifact = joblib.load(out / "model.joblib")
    explainer = explain.build_explainer(artifact["pipeline"], artifact["explainer_background"])

    frame = pd.DataFrame([Customer(**HIGH_RISK).model_dump()])
    drivers = explainer.drivers(frame)[0]
    assert len(drivers) == 3
    assert {d["feature"] for d in drivers} & {"tenure", "Contract"}
