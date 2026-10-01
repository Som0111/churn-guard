"""Feature engineering and preprocessing.

The engineered features and the encoders live inside a single scikit-learn
``Pipeline`` so that training and serving cannot drift apart: the API loads the
exact object that was fitted, raw customer dict in, probability out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churnguard import config

# Domain features added on top of the raw columns.
ENGINEERED_NUMERIC = [
    "avg_monthly_spend",
    "spend_vs_current_ratio",
    "tenure_years",
    "n_addon_services",
    "is_new_customer",
]

ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


# Raw customer fields each engineered feature is computed from (used to fold SHAP
# contributions back onto original fields - see explain.py).
DERIVED_SOURCES = {
    "avg_monthly_spend": ["TotalCharges", "tenure"],
    "spend_vs_current_ratio": ["MonthlyCharges", "TotalCharges", "tenure"],
    "tenure_years": ["tenure"],
    "n_addon_services": ADDON_SERVICES,
    "is_new_customer": ["tenure"],
}


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features a churn analyst would actually ask for.

    * ``avg_monthly_spend`` - lifetime spend normalised by tenure.
    * ``spend_vs_current_ratio`` - current bill against the historical
      average; a value above 1 means the customer was recently up-priced,
      which is a classic churn trigger.
    * ``n_addon_services`` - each extra service is another switching cost.
    """
    df = df.copy()

    safe_tenure = df["tenure"].clip(lower=1)
    df["avg_monthly_spend"] = df["TotalCharges"] / safe_tenure
    df["spend_vs_current_ratio"] = df["MonthlyCharges"] / df["avg_monthly_spend"].replace(
        0, np.nan
    )
    df["spend_vs_current_ratio"] = df["spend_vs_current_ratio"].fillna(1.0)
    df["tenure_years"] = df["tenure"] / 12.0
    df["n_addon_services"] = sum(
        (df[col] == "Yes").astype(int) for col in ADDON_SERVICES
    )
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numerics, impute + one-hot encode categoricals."""
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                config.NUMERIC_FEATURES + ENGINEERED_NUMERIC,
            ),
            ("categorical", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_stage() -> Pipeline:
    """Domain features followed by encoding - the shared front half."""
    return Pipeline(
        [
            (
                "domain",
                FunctionTransformer(add_domain_features, validate=False),
            ),
            ("preprocess", build_preprocessor()),
        ]
    )


def feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """Read the post-encoding column names out of a fitted pipeline."""
    preprocessor = fitted_pipeline.named_steps["features"].named_steps["preprocess"]
    return list(preprocessor.get_feature_names_out())
