"""Training entry point.

Trains three candidate models, selects on cross-validated ROC-AUC, tunes the
decision threshold on profit, then writes the fitted pipeline, a model card and
the report figures.

Run with:  python -m churnguard.train
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from churnguard import config, data, evaluate, features

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger("churnguard.train")


def candidate_models() -> dict[str, object]:
    """The three candidates, ordered from most to least interpretable.

    ``class_weight="balanced"`` handles the ~73/27 imbalance without resampling,
    which keeps the probabilities meaningful for the cost model downstream.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=24,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    """Feature stage + estimator, so one object serves end to end."""
    return Pipeline(
        [
            ("features", features.build_feature_stage()),
            ("model", estimator),
        ]
    )


def compare_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Cross-validate every candidate on the training split only."""
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    rows = []

    for name, estimator in candidate_models().items():
        started = time.perf_counter()
        scores = cross_validate(
            build_pipeline(estimator),
            X_train,
            y_train,
            cv=cv,
            scoring=["roc_auc", "average_precision", "f1"],
            n_jobs=-1,
        )
        rows.append(
            {
                "model": name,
                "cv_roc_auc": float(np.mean(scores["test_roc_auc"])),
                "cv_roc_auc_std": float(np.std(scores["test_roc_auc"])),
                "cv_pr_auc": float(np.mean(scores["test_average_precision"])),
                "cv_f1": float(np.mean(scores["test_f1"])),
                "fit_seconds": round(time.perf_counter() - started, 2),
            }
        )
        logger.info(
            "%-20s ROC-AUC %.4f (+/- %.4f)  PR-AUC %.4f",
            name,
            rows[-1]["cv_roc_auc"],
            rows[-1]["cv_roc_auc_std"],
            rows[-1]["cv_pr_auc"],
        )

    return pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)


def top_drivers(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, top_n: int = 12
) -> list[dict]:
    """Permutation importance - model agnostic, measured on held-out data."""
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=8,
        random_state=config.RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )
    ranked = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    return [
        {
            "feature": row.feature,
            "roc_auc_drop_when_shuffled": round(float(row.importance), 5),
            "std": round(float(row.std), 5),
        }
        for row in ranked.itertuples()
    ]


def plot_importance(drivers: list[dict]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(drivers).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        frame["feature"],
        frame["roc_auc_drop_when_shuffled"],
        xerr=frame["std"],
        color="#2563eb",
    )
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled")
    ax.set_title("What actually drives churn predictions")
    fig.tight_layout()
    path = config.FIGURE_DIR / "feature_importance.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return evaluate._repo_relative(path)


def main(skip_figures: bool = False) -> dict:
    config.ensure_dirs()

    logger.info("Loading data")
    df = data.load_clean()
    dataset_summary = data.summarize(df)
    logger.info("Dataset: %s", dataset_summary)

    X_train, X_test, y_train, y_test = data.split(df)
    logger.info("Train %d rows / test %d rows", len(X_train), len(X_test))

    logger.info("Cross-validating %d candidates", len(candidate_models()))
    leaderboard = compare_models(X_train, y_train)
    best_name = str(leaderboard.iloc[0]["model"])
    logger.info("Winner: %s", best_name)

    pipeline = build_pipeline(candidate_models()[best_name])
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    threshold_report = evaluate.optimize_threshold(y_test.to_numpy(), proba)
    threshold = threshold_report["optimal_threshold"]

    metrics_default = evaluate.classification_metrics(y_test.to_numpy(), proba, 0.5)
    metrics_tuned = evaluate.classification_metrics(y_test.to_numpy(), proba, threshold)

    logger.info(
        "Test ROC-AUC %.4f | tuned threshold %.2f | campaign profit $%s",
        metrics_tuned["roc_auc"],
        threshold,
        f"{threshold_report['net_benefit_optimal']:,.0f}",
    )

    logger.info("Computing permutation importance")
    drivers = top_drivers(pipeline, X_test, y_test)

    figures: list[str] = []
    if not skip_figures:
        figures = evaluate.generate_figures(y_test.to_numpy(), proba, threshold)
        figures.append(plot_importance(drivers))

    joblib.dump(
        {"pipeline": pipeline, "threshold": threshold, "model_name": best_name},
        config.MODEL_PATH,
    )
    logger.info("Model saved to %s", config.MODEL_PATH)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset_summary,
        "selected_model": best_name,
        "leaderboard": leaderboard.round(4).to_dict(orient="records"),
        "test_metrics_default_threshold": metrics_default,
        "test_metrics_tuned_threshold": metrics_tuned,
        "business_impact": threshold_report,
        "top_drivers": drivers,
        "figures": figures,
        "environment": {
            "python": platform.python_version(),
            "random_state": config.RANDOM_STATE,
        },
    }

    evaluate.save_metrics(report)
    config.METADATA_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ChurnGuard model.")
    parser.add_argument(
        "--skip-figures", action="store_true", help="skip matplotlib rendering"
    )
    args = parser.parse_args()
    main(skip_figures=args.skip_figures)
