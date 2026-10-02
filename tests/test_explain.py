"""SHAP drivers: grouped onto original fields, additive, and safe to lose."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import synthetic_customers
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from test_api import HIGH_RISK, LOW_RISK

from churnguard import api, config, explain, features
from churnguard.train import build_pipeline

DIRECTIONS = {"raises", "lowers"}


@pytest.fixture(scope="module")
def data():
    return synthetic_customers(n=700)


def _fit(estimator, data):
    X, y = data
    model = build_pipeline(estimator).fit(X, y)
    return model, explain.make_background(model, X)


def test_every_transformed_feature_maps_to_original_fields(data):
    model, _ = _fit(LogisticRegression(max_iter=500), data)
    for name in features.feature_names(model):
        sources = explain.source_columns(name)
        assert sources and set(sources) <= set(config.FEATURES), name


def test_every_engineered_feature_has_a_declared_source():
    assert set(features.DERIVED_SOURCES) == set(features.ENGINEERED_NUMERIC)
    for sources in features.DERIVED_SOURCES.values():
        assert set(sources) <= set(config.FEATURES)


def test_tenure_derived_features_fold_into_tenure(data):
    assert explain.source_columns("tenure_years") == ["tenure"]
    assert explain.source_columns("is_new_customer") == ["tenure"]
    assert explain.source_columns("Contract_Two year") == ["Contract"]
    assert explain.source_columns("PaymentMethod_Mailed check") == ["PaymentMethod"]


def test_grouped_contributions_sum_to_the_models_total(data):
    """Grouping must only re-label SHAP values, never create or lose any."""
    X, _ = data
    model, background = _fit(LogisticRegression(max_iter=500), data)
    explainer = explain.build_explainer(model, background)
    frame = X.iloc[:25]

    grouped = explainer.contributions(frame)
    transformed = model.named_steps["features"].transform(frame)
    raw = explainer._explainer.shap_values(transformed)
    assert grouped.shape == (25, len(config.FEATURES))
    np.testing.assert_allclose(grouped.sum(axis=1), raw.sum(axis=1), atol=1e-9)

    # and they explain the score: base value + contributions = the model's log-odds
    logit = model.decision_function(frame)
    np.testing.assert_allclose(
        grouped.sum(axis=1) + explainer._explainer.expected_value, logit, atol=1e-6
    )


def test_drivers_shape_and_ordering(data):
    X, _ = data
    model, background = _fit(LogisticRegression(max_iter=500), data)
    drivers = explain.build_explainer(model, background).drivers(X.iloc[:10])
    assert len(drivers) == 10
    for reasons in drivers:
        assert len(reasons) == 3
        assert len({d["feature"] for d in reasons}) == 3
        assert all(d["feature"] in config.FEATURES for d in reasons)
        assert all(d["direction"] in DIRECTIONS and d["magnitude"] >= 0 for d in reasons)
        magnitudes = [d["magnitude"] for d in reasons]
        assert magnitudes == sorted(magnitudes, reverse=True)


def test_random_forest_uses_the_tree_explainer(data):
    X, _ = data
    model, background = _fit(RandomForestClassifier(n_estimators=20, random_state=0), data)
    explainer = explain.build_explainer(model, background)
    assert type(explainer._explainer).__name__ == "TreeExplainer"
    reasons = explainer.drivers(X.iloc[:3])
    assert all(len(r) == 3 for r in reasons)


def test_unsupported_estimator_disables_drivers_instead_of_failing(data):
    model, background = _fit(HistGradientBoostingClassifier(max_iter=20), data)
    assert explain.build_explainer(model, background) is None


def test_missing_background_disables_drivers(data):
    model, _ = _fit(LogisticRegression(max_iter=500), data)
    assert explain.build_explainer(model, None) is None


# --------------------------------------------------------------------------- #
# Through the API
# --------------------------------------------------------------------------- #
def test_predict_returns_three_valid_drivers(client):
    body = client.post("/predict", json=HIGH_RISK).json()
    drivers = body["top_drivers"]
    assert len(drivers) == 3
    assert all(d["feature"] in config.FEATURES for d in drivers)
    assert all(d["direction"] in DIRECTIONS for d in drivers)


def test_new_month_to_month_fiber_customer_is_explained_by_tenure_or_contract(client):
    drivers = client.post("/predict", json=HIGH_RISK).json()["top_drivers"]
    named = {d["feature"] for d in drivers}
    assert named & {"tenure", "Contract"}, named
    # these fields push the risk up for this customer
    assert all(d["direction"] == "raises" for d in drivers if d["feature"] in {"tenure", "Contract"})


def test_a_loyal_customer_gets_risk_lowering_reasons(client):
    drivers = client.post("/predict", json=LOW_RISK).json()["top_drivers"]
    assert any(d["direction"] == "lowers" for d in drivers)


def test_batch_returns_drivers_per_customer_in_order(client):
    results = client.post("/predict/batch", json={"customers": [HIGH_RISK, LOW_RISK]}).json()
    assert [len(r["top_drivers"]) for r in results] == [3, 3]
    assert results[0]["top_drivers"] != results[1]["top_drivers"]


def test_scoring_still_works_when_the_explainer_is_unavailable(client, monkeypatch):
    monkeypatch.setitem(api.MODEL, "explainer", None)
    response = client.post("/predict", json=HIGH_RISK)
    assert response.status_code == 200
    assert response.json()["top_drivers"] is None
    assert 0 <= response.json()["churn_probability"] <= 1


def test_a_failing_explainer_does_not_fail_the_score(client, monkeypatch):
    class Boom:
        def drivers(self, frame):
            raise RuntimeError("shap exploded")

    monkeypatch.setitem(api.MODEL, "explainer", Boom())
    response = client.post("/predict", json=HIGH_RISK)
    assert response.status_code == 200 and response.json()["top_drivers"] is None


# --------------------------------------------------------------------------- #
# Permutation importance: linked fields must be shuffled together
# --------------------------------------------------------------------------- #
def test_shuffling_tenure_alone_creates_impossible_rows_and_overstates_importance(data):
    from churnguard.train import PERMUTATION_GROUPS, permutation_drop

    X, y = data
    model, _ = _fit(LogisticRegression(max_iter=500), data)
    alone, _ = permutation_drop(model, X, y, ["tenure"], n_repeats=5)
    joint, _ = permutation_drop(model, X, y, PERMUTATION_GROUPS["tenure + TotalCharges"], n_repeats=5)
    assert joint < alone  # the joint shuffle does not manufacture impossible customers


def test_permutation_drop_is_seeded_and_positive_for_an_informative_field(data):
    from churnguard.train import permutation_drop

    X, y = data
    model, _ = _fit(LogisticRegression(max_iter=500), data)
    a = permutation_drop(model, X, y, ["Contract"], n_repeats=4)
    assert a == permutation_drop(model, X, y, ["Contract"], n_repeats=4)
    assert a[0] > 0


def test_top_drivers_reports_the_linked_block_as_one_unit(data):
    from churnguard.train import top_drivers

    X, y = data
    model, _ = _fit(LogisticRegression(max_iter=500), data)
    names = [d["feature"] for d in top_drivers(model, X, y, top_n=30)]
    assert "tenure + TotalCharges" in names
    assert "tenure" not in names and "TotalCharges" not in names
