"""Bad data and bad config must fail early with a readable message."""

from __future__ import annotations

import typing

import pandas as pd
import pytest

from churnguard import config, data
from churnguard.api import Customer
from churnguard.config import CostModel


@pytest.fixture
def raw() -> pd.DataFrame:
    return data.load_raw().head(50).copy()


def test_valid_raw_data_passes(raw):
    data.validate_raw(raw)


def test_missing_column_is_named(raw):
    with pytest.raises(data.DataValidationError, match="Missing required column.*Contract"):
        data.validate_raw(raw.drop(columns=["Contract"]))


def test_bad_target_value(raw):
    raw.loc[raw.index[0], config.TARGET] = "Maybe"
    with pytest.raises(data.DataValidationError, match=r"Churn.*Maybe"):
        data.validate_raw(raw)


def test_malformed_numeric_column(raw):
    raw["tenure"] = raw["tenure"].astype(object)
    raw.loc[raw.index[3], "tenure"] = "twelve"
    with pytest.raises(data.DataValidationError, match=r"tenure.*non-numeric.*twelve"):
        data.validate_raw(raw)


def test_blank_total_charges_is_still_allowed(raw):
    raw["TotalCharges"] = raw["TotalCharges"].astype(object)
    raw.loc[raw.index[0], "TotalCharges"] = " "
    data.validate_raw(raw)


def test_unknown_category(raw):
    raw.loc[raw.index[1], "Contract"] = "Three year"
    with pytest.raises(data.DataValidationError, match=r"Contract.*Three year"):
        data.validate_raw(raw)


def test_out_of_range_numeric(raw):
    raw.loc[raw.index[2], "MonthlyCharges"] = -5.0
    with pytest.raises(data.DataValidationError, match=r"MonthlyCharges.*outside"):
        data.validate_raw(raw)


def test_null_id(raw):
    raw.loc[raw.index[0], config.ID_COLUMN] = None
    with pytest.raises(data.DataValidationError, match="null id"):
        data.validate_raw(raw)


def test_every_problem_is_reported_at_once(raw):
    raw.loc[raw.index[0], "Contract"] = "Bad"
    raw.loc[raw.index[1], config.TARGET] = "Bad"
    with pytest.raises(data.DataValidationError) as exc:
        data.validate_raw(raw)
    assert "Contract" in str(exc.value) and "Churn" in str(exc.value)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"offer_cost": -1}, "offer_cost"),
        ({"customer_lifetime_value": -0.01}, "customer_lifetime_value"),
        ({"offer_success_rate": -0.1}, "offer_success_rate"),
        ({"offer_success_rate": 1.1}, "offer_success_rate"),
        ({"offer_cost": float("nan")}, "offer_cost"),
        ({"offer_success_rate": float("nan")}, "offer_success_rate"),
    ],
)
def test_invalid_cost_model_raises(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CostModel(**kwargs)


def test_cost_model_boundaries_are_valid():
    CostModel(offer_cost=0, customer_lifetime_value=0, offer_success_rate=0)
    CostModel(offer_success_rate=1)


def test_api_categories_match_config():
    """The API's Literal types and the data check must agree on what is allowed."""
    for field, allowed in config.ALLOWED_CATEGORIES.items():
        assert set(typing.get_args(Customer.model_fields[field].annotation)) == set(allowed), field
