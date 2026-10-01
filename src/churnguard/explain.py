"""Per-customer SHAP drivers, reported against the original customer fields.

SHAP runs on the model's *transformed* inputs (scaled numerics, one-hot columns,
engineered features). Those are meaningless to a retention analyst, so each
contribution is folded back onto the raw field(s) it was built from:

* one-hot columns -> their categorical field (``Contract_Two year`` -> ``Contract``)
* engineered features -> their source fields, per ``features.DERIVED_SOURCES``
  (``tenure_years``, ``is_new_customer`` -> ``tenure``)
* a feature built from several fields splits its contribution equally between
  them (``avg_monthly_spend`` -> half ``TotalCharges``, half ``tenure``)

Grouping only re-labels: the grouped values still sum to the model's total
contribution. Units are log-odds of churn relative to the training average.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from churnguard import config, features

logger = logging.getLogger(__name__)


def base_pipeline(model):
    """The fitted feature+estimator Pipeline inside a (possibly calibrated) model."""
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model


def make_background(model, X_train: pd.DataFrame, n: int = 200, seed: int = config.RANDOM_STATE):
    """A transformed training sample that SHAP measures contributions against."""
    sample = X_train.sample(min(n, len(X_train)), random_state=seed)
    return base_pipeline(model).named_steps["features"].transform(sample)


def source_columns(name: str) -> list[str]:
    """Raw customer field(s) that a transformed feature was built from."""
    if name in config.NUMERIC_FEATURES:
        return [name]
    if name in features.DERIVED_SOURCES:
        return features.DERIVED_SOURCES[name]
    for column in config.CATEGORICAL_FEATURES:
        if name.startswith(f"{column}_"):
            return [column]
    raise ValueError(f"No source column known for transformed feature {name!r}")


class DriverExplainer:
    def __init__(self, model, shap_explainer):
        self._pipeline = base_pipeline(model)
        self._explainer = shap_explainer
        names = features.feature_names(self._pipeline)
        # grouping[i, j] = share of transformed feature i credited to raw field j
        self._fields = list(config.FEATURES)
        self._grouping = np.zeros((len(names), len(self._fields)))
        for i, name in enumerate(names):
            sources = source_columns(name)
            for source in sources:
                self._grouping[i, self._fields.index(source)] = 1 / len(sources)

    def contributions(self, frame: pd.DataFrame) -> np.ndarray:
        """Grouped SHAP values, shape (n_customers, n_raw_fields), in log-odds."""
        transformed = self._pipeline.named_steps["features"].transform(frame)
        values = self._explainer.shap_values(transformed)
        if isinstance(values, list):  # older shap: one array per class
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:  # newer shap: (n, features, classes)
            values = values[..., 1]
        return values @ self._grouping

    def drivers(self, frame: pd.DataFrame, top_n: int = 3) -> list[list[dict]]:
        out = []
        for row in self.contributions(frame):
            top = np.argsort(-np.abs(row))[:top_n]
            out.append(
                [
                    {
                        "feature": self._fields[j],
                        "direction": "raises" if row[j] > 0 else "lowers",
                        "magnitude": round(float(abs(row[j])), 4),
                    }
                    for j in top
                ]
            )
        return out


def build_explainer(model, background) -> DriverExplainer | None:
    """SHAP explainer for the fitted model, or None if it cannot be built.

    None (with a logged reason) lets the API keep scoring without drivers when
    ``shap`` is not installed or the estimator type is unsupported.
    """
    if background is None:
        logger.warning("No explainer background in the model artifact; drivers disabled")
        return None
    try:
        import shap
    except ImportError:
        logger.warning("shap is not installed (pip install '.[explain]'); drivers disabled")
        return None

    estimator = base_pipeline(model).named_steps["model"]
    if isinstance(estimator, LogisticRegression):
        shap_explainer = shap.LinearExplainer(estimator, background)
    elif isinstance(estimator, RandomForestClassifier):
        shap_explainer = shap.TreeExplainer(estimator)
    else:
        logger.warning("No SHAP explainer for %s; drivers disabled", type(estimator).__name__)
        return None
    return DriverExplainer(model, shap_explainer)
