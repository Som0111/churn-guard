"""The model report must name the exact data, code and library versions."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from churnguard import config
from churnguard.train import build_pipeline, build_provenance, candidate_models

REQUIRED = {
    "model_version", "git_sha", "git_dirty", "trained_at", "dataset", "estimator",
    "estimator_hyperparameters", "calibration_method", "threshold_method", "splits",
    "dependencies", "python", "random_state",
}


def test_provenance_has_every_field():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            **{c: rng.uniform(1, 50, 60) for c in config.NUMERIC_FEATURES},
            **{c: rng.choice(["Yes", "No"], 60) for c in config.CATEGORICAL_FEATURES},
        }
    )
    y = pd.Series(rng.binomial(1, 0.3, 60))
    model = build_pipeline(candidate_models()["logistic_regression"]).fit(X, y)

    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    prov = build_provenance(model, "no_class_weight", {"train": y, "validation": y, "test": y}, when)

    assert REQUIRED <= prov.keys()
    assert prov["model_version"] == f"{prov['git_sha']}-20260102T030405Z"
    assert prov["dataset"]["sha256"] == config.DATA_SHA256 and prov["dataset"]["source_url"]
    assert prov["estimator"] == "LogisticRegression"
    assert set(prov["splits"]["train"]) == {"n", "churn_rate"}
    assert {"numpy", "pandas", "scikit-learn", "scipy", "joblib"} <= prov["dependencies"].keys()
