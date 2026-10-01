"""Survival analysis: runs end to end on the cleaned data and says something sensible."""

from __future__ import annotations

import json
import sys

import pytest

from churnguard import config, data, survival


@pytest.fixture(scope="module")
def result():
    return survival.run(write_figures=False)


def _km(result, column):
    return next(k for k in result["kaplan_meier"] if k["column"] == column)


def test_runs_end_to_end_on_cleaned_data(result):
    assert result["n_customers"] + result["n_dropped_zero_tenure"] == len(data.load_clean())
    assert result["n_dropped_zero_tenure"] == 11  # the documented brand-new customers
    assert {k["column"] for k in result["kaplan_meier"]} == {"Contract", "InternetService"}


def test_validation_concordance_is_stored_and_above_chance(result):
    cox = result["cox"]
    assert 0.5 < cox["concordance_validation"] <= 1.0
    assert 0.5 < cox["concordance_train"] <= 1.0


def test_test_split_is_never_used(result):
    total = result["n_customers"]
    used = result["cox"]["n_train"] + result["cox"]["n_validation"]
    assert used == pytest.approx(0.8 * total, rel=0.01)  # train + validation only


def test_kaplan_meier_curves_are_valid_and_ordered_by_contract(result):
    groups = _km(result, "Contract")["groups"]
    for group in groups.values():
        values = [group["survival_at_months"][str(t)] for t in survival.HORIZONS]
        assert all(0 <= v <= 1 for v in values)
        assert values == sorted(values, reverse=True)  # survival never increases
    at_12 = {name: g["survival_at_months"]["12"] for name, g in groups.items()}
    assert at_12["Month-to-month"] < at_12["One year"] <= at_12["Two year"]
    assert _km(result, "Contract")["logrank_p_value"] < 0.001


def test_cox_hazard_ratios_point_the_expected_way(result):
    hr = {k: v["hazard_ratio"] for k, v in result["cox"]["hazard_ratios"].items()}
    assert hr["Contract_Two year"] < hr["Contract_One year"] < 1.0
    assert hr["InternetService_Fiber optic"] > 1.0
    for h in result["cox"]["hazard_ratios"].values():
        assert h["ci_low"] <= h["hazard_ratio"] <= h["ci_high"]


def test_total_charges_is_kept_out_of_the_covariates(result):
    """TotalCharges ~ tenure x monthly charge, so it would leak the clock."""
    assert not any("TotalCharges" in name for name in result["cox"]["hazard_ratios"])


def test_proportional_hazards_is_checked_for_every_covariate(result):
    ph = result["cox"]["proportional_hazards"]
    assert set(ph["p_values"]) == set(result["cox"]["hazard_ratios"])
    assert set(ph["violations"]) <= set(ph["p_values"])
    assert all(ph["p_values"][name] < ph["alpha"] for name in ph["violations"])


def test_main_writes_the_reports_and_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURE_DIR", tmp_path)
    monkeypatch.setattr(survival, "SUMMARY_JSON", tmp_path / "survival_summary.json")
    monkeypatch.setattr(survival, "SUMMARY_MD", tmp_path / "survival_summary.md")

    result = survival.main()

    saved = json.loads((tmp_path / "survival_summary.json").read_text(encoding="utf-8"))
    assert saved["cox"]["concordance_validation"] == result["cox"]["concordance_validation"] > 0.5
    markdown = (tmp_path / "survival_summary.md").read_text(encoding="utf-8")
    for heading in ("Hazard vs probability", "Kaplan-Meier", "Cox proportional-hazards", "proportional hazards"):
        assert heading in markdown
    assert (tmp_path / "survival_km.png").exists()
    assert (tmp_path / "survival_cox_hazard_ratios.png").exists()


def test_a_missing_lifelines_explains_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "lifelines", None)
    with pytest.raises(RuntimeError, match=r"\.\[analysis\]"):
        survival.run(write_figures=False)
