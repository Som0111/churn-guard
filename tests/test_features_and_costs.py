"""Feature engineering and the business cost model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnguard import features
from churnguard.config import CostModel
from churnguard.evaluate import (
    apply_threshold,
    classification_metrics,
    expected_calibration_error,
    optimize_threshold,
    profit_curve,
    reliability_table,
)


@pytest.fixture
def sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenure": [0, 12, 24],
            "MonthlyCharges": [70.0, 50.0, 100.0],
            "TotalCharges": [0.0, 600.0, 2400.0],
            "OnlineSecurity": ["No", "Yes", "Yes"],
            "OnlineBackup": ["No", "Yes", "No"],
            "DeviceProtection": ["No", "No", "Yes"],
            "TechSupport": ["No", "Yes", "Yes"],
            "StreamingTV": ["No", "No", "Yes"],
            "StreamingMovies": ["No", "No", "Yes"],
        }
    )


def test_domain_features_are_added(sample):
    out = features.add_domain_features(sample)
    for column in features.ENGINEERED_NUMERIC:
        assert column in out.columns


def test_zero_tenure_does_not_divide_by_zero(sample):
    out = features.add_domain_features(sample)
    assert np.isfinite(out["avg_monthly_spend"]).all()
    assert np.isfinite(out["spend_vs_current_ratio"]).all()


def test_addon_services_are_counted(sample):
    out = features.add_domain_features(sample)
    # row 0: none. row 1: backup + support + security. row 2: all but backup.
    assert out["n_addon_services"].tolist() == [0, 3, 5]


def test_spend_ratio_flags_a_recent_price_rise(sample):
    """Customer 1 pays 50/mo now but averaged 50 historically -> ratio 1.0."""
    out = features.add_domain_features(sample)
    assert out.loc[1, "spend_vs_current_ratio"] == pytest.approx(1.0)


def test_cost_model_arithmetic():
    costs = CostModel(offer_cost=50, customer_lifetime_value=500, offer_success_rate=0.3)
    assert costs.true_positive_value == pytest.approx(100.0)
    assert costs.false_positive_value == pytest.approx(-50.0)
    # 10 caught churners, 4 wasted offers
    assert costs.net_benefit(tp=10, fp=4) == pytest.approx(10 * 100 - 4 * 50)


def test_optimal_threshold_beats_the_default():
    """A perfect ranker should be tuned to at least match the 0.5 default."""
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.27, size=2000)
    # Well-separated but imperfect scores.
    proba = np.clip(y_true * 0.55 + rng.normal(0.2, 0.18, size=2000), 0.001, 0.999)

    report = optimize_threshold(y_true, proba)
    assert 0 < report["optimal_threshold"] < 1
    assert report["net_benefit_optimal"] >= report["net_benefit_at_0.5"]
    assert report["uplift_vs_default"] >= 0


def test_threshold_between_old_grid_points_is_found():
    """Best cut sits at 0.5004 - inside the old 0.01-0.99 / 200-step grid cell."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.10, 0.20, 0.30, 0.5004, 0.80, 0.90])
    report = optimize_threshold(y_true, proba)
    assert report["optimal_threshold"] == pytest.approx(0.5004)
    old_grid = np.linspace(0.01, 0.99, 200)
    assert not np.isclose(old_grid, 0.5004, atol=1e-6).any()
    assert report["churners_caught"] == 3 and report["wasted_offers"] == 0


def test_test_labels_never_enter_threshold_search():
    """Flipping every 'test' label must not change the validation-tuned threshold."""
    rng = np.random.default_rng(3)
    y_val = rng.binomial(1, 0.3, size=400)
    p_val = rng.uniform(size=400)
    threshold = optimize_threshold(y_val, p_val)["optimal_threshold"]

    p_test = rng.uniform(size=300)
    a = apply_threshold(rng.binomial(1, 0.3, size=300), p_test, threshold)
    b = apply_threshold(np.ones(300, dtype=int), p_test, threshold)
    assert a["optimal_threshold"] == b["optimal_threshold"] == threshold


def test_tuned_threshold_never_loses_to_default_on_validation():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        y = rng.binomial(1, 0.27, size=500)
        p = np.clip(y * 0.3 + rng.uniform(size=500) * 0.7, 0, 1)
        report = optimize_threshold(y, p)
        assert report["net_benefit_optimal"] >= report["net_benefit_at_0.5"]


def test_ece_is_zero_for_calibrated_and_large_for_overconfident():
    rng = np.random.default_rng(5)
    p = rng.uniform(size=20000)
    y = rng.binomial(1, p)
    assert expected_calibration_error(y, p) < 0.02
    assert expected_calibration_error(y, np.clip(p + 0.3, 0, 1)) > 0.15
    table = reliability_table(y, p)
    assert len(table) == 10 and sum(r["n"] for r in table) == 20000


def test_classification_metrics_include_ece():
    y = np.array([0, 0, 1, 1] * 10)
    p = np.array([0.1, 0.4, 0.6, 0.9] * 10)
    assert 0 <= classification_metrics(y, p, 0.5)["ece"] <= 1


def test_profit_curve_is_monotone_in_targeting():
    """Raising the threshold can only shrink the targeted population."""
    rng = np.random.default_rng(1)
    y_true = rng.binomial(1, 0.3, size=500)
    proba = rng.uniform(size=500)

    curve = profit_curve(y_true, proba)
    targeted = curve["customers_targeted"].to_numpy()
    assert (np.diff(targeted) <= 0).all()


def test_engineered_features_exact_values():
    """Hand-computed expectations for every engineered column."""
    df = pd.DataFrame(
        {
            "tenure": [0, 6, 7, 24],
            "MonthlyCharges": [70.0, 90.0, 60.0, 100.0],
            "TotalCharges": [0.0, 300.0, 350.0, 2400.0],
            "OnlineSecurity": ["No", "Yes", "Yes", "Yes"],
            "OnlineBackup": ["No", "Yes", "No", "Yes"],
            "DeviceProtection": ["No", "No", "No", "Yes"],
            "TechSupport": ["No", "No", "Yes", "Yes"],
            "StreamingTV": ["No", "No", "No", "Yes"],
            "StreamingMovies": ["No", "No", "No", "Yes"],
        }
    )
    out = features.add_domain_features(df)

    # tenure 0 is clipped to 1 month, so its total (0) gives an average of 0
    assert out["avg_monthly_spend"].tolist() == pytest.approx([0.0, 50.0, 50.0, 100.0])
    # average 0 is undefined -> falls back to 1.0; otherwise monthly / average
    assert out["spend_vs_current_ratio"].tolist() == pytest.approx([1.0, 1.8, 1.2, 1.0])
    assert out["tenure_years"].tolist() == pytest.approx([0.0, 0.5, 7 / 12, 2.0])
    assert out["n_addon_services"].tolist() == [0, 2, 2, 6]
    assert out["is_new_customer"].tolist() == [1, 1, 0, 0]  # tenure <= 6 months


def test_single_class_metrics_fail_loudly():
    with pytest.raises(ValueError, match="both classes"):
        classification_metrics(np.zeros(10, dtype=int), np.linspace(0.1, 0.9, 10))


def test_single_class_profit_is_still_defined():
    """All stayers: any campaign only wastes money, so the best move is to do nothing."""
    y = np.zeros(50, dtype=int)
    p = np.linspace(0.05, 0.95, 50)
    report = optimize_threshold(y, p)
    assert report["net_benefit_optimal"] == 0.0 and report["customers_targeted"] == 0


def test_all_identical_predictions():
    y = np.array([0, 1] * 20)
    p = np.full(40, 0.3)
    assert classification_metrics(y, p, 0.5)["roc_auc"] == pytest.approx(0.5)
    assert expected_calibration_error(y, p) == pytest.approx(0.2)  # |0.3 - 0.5|
    report = optimize_threshold(y, p)
    # Constant scores can only target everyone or no one.
    assert report["net_benefit_optimal"] == max(0.0, report["net_benefit_blanket_campaign"])


def test_threshold_extremes_and_ties():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.4, 0.4, 0.9])
    assert apply_threshold(y, p, 0.0)["customers_targeted"] == 4  # everyone
    assert apply_threshold(y, p, 1.0)["customers_targeted"] == 0  # no one (p < 1)
    assert apply_threshold(y, p, 0.4)["customers_targeted"] == 3  # >= includes the tie
    assert apply_threshold(y, p, 0.4000001)["customers_targeted"] == 1
