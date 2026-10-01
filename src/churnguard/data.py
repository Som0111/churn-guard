"""Dataset download, cleaning and splitting.

Source: IBM Telco Customer Churn (7,043 customers, 21 columns).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churnguard import config

logger = logging.getLogger(__name__)


class DatasetIntegrityError(RuntimeError):
    """The raw CSV does not match the pinned SHA-256."""


def sha256_of(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def verify_checksum(content: bytes, source: str | Path) -> str:
    """Return the SHA-256 of ``content`` or raise if it is not the pinned dataset."""
    digest = sha256_of(content)
    if digest != config.DATA_SHA256:
        raise DatasetIntegrityError(
            f"{source}: SHA-256 {digest} does not match the pinned "
            f"{config.DATA_SHA256}. Delete the file and re-download, or update "
            "config.DATA_SHA256 deliberately if the dataset was meant to change."
        )
    return digest


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
    verify_checksum(response.content, config.DATA_URL)  # never cache a bad file
    config.RAW_CSV.write_bytes(response.content)
    logger.info("Saved %s KB", len(response.content) // 1024)


def load_raw() -> pd.DataFrame:
    """Load the cached CSV (downloading it first if necessary), checksum-verified."""
    if not config.RAW_CSV.exists():
        download()
    verify_checksum(config.RAW_CSV.read_bytes(), config.RAW_CSV)
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
    """Stratified 60/20/20 train / validation / test split on the churn label.

    train: model selection (CV). validation: threshold tuning. test: scored once.
    """
    X = df[config.FEATURES]
    y = df[config.TARGET]
    X_rest, X_test, y_rest, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=config.VAL_SIZE / (1 - config.TEST_SIZE),
        random_state=config.RANDOM_STATE,
        stratify=y_rest,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def summarize(df: pd.DataFrame) -> dict:
    """Headline numbers used by the README and the EDA report."""
    churn_rate = float(df[config.TARGET].mean())
    return {
        "n_rows": len(df),
        "n_features": len(config.FEATURES),
        "churn_rate": round(churn_rate, 4),
        "class_imbalance_ratio": round((1 - churn_rate) / churn_rate, 2),
        "monthly_revenue_at_risk": round(
            float(df.loc[df[config.TARGET] == 1, "MonthlyCharges"].sum()), 2
        ),
    }
