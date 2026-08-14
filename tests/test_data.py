"""Data-layer contracts: the cleaning step must fix the known quirks."""

from __future__ import annotations

import pandas as pd
import pytest

from churnguard import config, data


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    return data.load_clean()


def test_target_is_binary(clean_df):
    assert set(clean_df[config.TARGET].unique()) == {0, 1}


def test_total_charges_is_numeric_with_no_gaps(clean_df):
    assert pd.api.types.is_numeric_dtype(clean_df["TotalCharges"])
    assert clean_df["TotalCharges"].isna().sum() == 0


def test_zero_tenure_customers_get_zero_total_charges():
    """The 11 blank TotalCharges rows are brand-new customers, not missing data."""
    raw = data.load_raw()
    blanks = raw["TotalCharges"].astype(str).str.strip() == ""
    assert blanks.sum() > 0, "fixture drifted: expected blank TotalCharges in raw data"
    assert (raw.loc[blanks, "tenure"] == 0).all()

    cleaned = data.clean(raw)
    assert (cleaned.loc[cleaned["tenure"] == 0, "TotalCharges"] == 0).all()


def test_no_duplicate_customers(clean_df):
    assert clean_df[config.ID_COLUMN].is_unique


def test_all_declared_features_exist(clean_df):
    missing = set(config.FEATURES) - set(clean_df.columns)
    assert not missing, f"declared but absent: {missing}"


def test_split_is_stratified_and_leak_free(clean_df):
    X_train, X_test, y_train, y_test = data.split(clean_df)

    assert len(X_train) + len(X_test) == len(clean_df)
    assert abs(y_train.mean() - y_test.mean()) < 0.02

    # The target and the ID must never reach the model.
    assert config.TARGET not in X_train.columns
    assert config.ID_COLUMN not in X_train.columns
