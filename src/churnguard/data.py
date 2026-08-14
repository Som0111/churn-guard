"""Dataset download, cleaning and splitting.

Source: IBM Telco Customer Churn (7,043 customers, 21 columns).
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from churnguard import config

logger = logging.getLogger(__name__)


def download(force: bool = False) -> None:
    """Fetch the raw CSV once and cache it under ``data/raw``."""
    config.ensure_dirs()
    if config.RAW_CSV.exists() and not force:
        logger.info("Dataset already cached at %s", config.RAW_CSV)
        return

    import requests

    logger.info("Downloading dataset from %s", config.DATA_URL)
    response = requests.get(config.DATA_URL, timeout=60)
    response.raise_for_status()
    config.RAW_CSV.write_bytes(response.content)
    logger.info("Saved %s KB", len(response.content) // 1024)


def load_raw() -> pd.DataFrame:
    """Load the cached CSV, downloading it first if necessary."""
    if not config.RAW_CSV.exists():
        download()
    return pd.read_csv(config.RAW_CSV)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the dataset's known quirks and encode the target as 0/1.

    ``TotalCharges`` ships as a string and holds a blank for the 11 customers
    whose tenure is 0 - they were billed for the first time after the snapshot
    was taken, so their true total is 0 rather than missing.
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    brand_new = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[brand_new, "TotalCharges"] = 0.0

    df = df.dropna(subset=["TotalCharges"])
    df = df.drop_duplicates(subset=[config.ID_COLUMN])

    df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)

    return df.reset_index(drop=True)


def load_clean() -> pd.DataFrame:
    """Convenience wrapper: raw CSV in, analysis-ready frame out."""
    return clean(load_raw())


def split(df: pd.DataFrame):
    """Stratified train/test split on the churn label."""
    X = df[config.FEATURES]
    y = df[config.TARGET]
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def summarize(df: pd.DataFrame) -> dict:
    """Headline numbers used by the README and the EDA report."""
    churn_rate = float(df[config.TARGET].mean())
    return {
        "n_rows": int(len(df)),
        "n_features": len(config.FEATURES),
        "churn_rate": round(churn_rate, 4),
        "class_imbalance_ratio": round((1 - churn_rate) / churn_rate, 2),
        "monthly_revenue_at_risk": round(
            float(df.loc[df[config.TARGET] == 1, "MonthlyCharges"].sum()), 2
        ),
    }
