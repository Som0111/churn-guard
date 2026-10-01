"""Drift monitoring: has the data the model sees moved away from what it was built on?

Detect and alert only - nothing here retrains or changes the model.

The drift verdict uses plain two-sample tests from SciPy (Kolmogorov-Smirnov for
numeric fields, chi-square for categorical ones), so the API can serve ``/drift``
without any heavy dependency. Evidently is optional (``pip install '.[monitor]'``)
and only renders the HTML report.

Reference = a sample of the *validation* split saved at train time, with the
model's own predictions on it. Scoring against validation rather than train
avoids the model's slight over-fit to its training rows.

Run the demo:  python -m churnguard.drift --demo
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from scipy import stats

from churnguard import config

logger = logging.getLogger(__name__)

PREDICTION = "churn_probability"
# SeniorCitizen is a 0/1 flag: a categorical test is the right one for it.
CATEGORICAL_FIELDS = [*config.CATEGORICAL_FEATURES, "SeniorCitizen"]
NUMERIC_FIELDS = [c for c in config.NUMERIC_FEATURES if c != "SeniorCitizen"]
MISSING = "<missing>"


def build_reference(model, X_ref: pd.DataFrame) -> pd.DataFrame:
    """Raw feature rows plus the model's predictions on them."""
    reference = X_ref[config.FEATURES].reset_index(drop=True).copy()
    reference[PREDICTION] = model.predict_proba(X_ref[config.FEATURES])[:, 1]
    return reference


def prepare(records: pd.DataFrame) -> pd.DataFrame:
    """Coerce arbitrary incoming records to the model's raw schema, keeping the damage visible.

    Missing columns become all-missing, extra columns are dropped, and numeric
    fields that do not parse become NaN (counted as missing, not silently fixed).
    """
    frame = records.reindex(columns=config.FEATURES).copy()
    for column in config.NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def data_quality(current: pd.DataFrame) -> dict:
    """Missingness per field and the share of values outside the known categories."""
    missing = current[config.FEATURES].isna().mean().round(4)
    unknown, unknown_cells, filled_cells = {}, 0, 0
    for column, allowed in config.ALLOWED_CATEGORIES.items():
        values = current[column].dropna().astype(str)
        bad = int((~values.isin(allowed)).sum())
        unknown[column] = round(bad / len(current), 4)
        unknown_cells += bad
        filled_cells += len(values)
    return {
        "missing_rate": {c: float(v) for c, v in missing.items() if v > 0},
        "overall_missing_rate": round(float(current[config.FEATURES].isna().to_numpy().mean()), 4),
        "unknown_category_rate": {c: v for c, v in unknown.items() if v > 0},
        "overall_unknown_category_rate": round(unknown_cells / max(filled_cells, 1), 4),
    }


def _numeric_test(ref: pd.Series, cur: pd.Series) -> dict:
    ref, cur = ref.dropna(), cur.dropna()
    if len(cur) < 2:  # a numeric field that vanished is drift, not "no evidence"
        return {"test": "ks", "statistic": 1.0, "p_value": 0.0}
    result = stats.ks_2samp(ref, cur)
    return {"test": "ks", "statistic": float(result.statistic), "p_value": float(result.pvalue)}


def _categorical_test(ref: pd.Series, cur: pd.Series) -> dict:
    table = pd.concat(
        [
            ref.astype(object).fillna(MISSING).astype(str).value_counts(),
            cur.astype(object).fillna(MISSING).astype(str).value_counts(),
        ],
        axis=1,
    ).fillna(0)
    if len(table) < 2:  # one category on both sides: identical by construction
        return {"test": "chi2", "statistic": 0.0, "p_value": 1.0}
    statistic, p_value, *_ = stats.chi2_contingency(table.to_numpy())
    return {"test": "chi2", "statistic": float(statistic), "p_value": float(p_value)}


def status_for(share_drifted: float, prediction_drifted: bool) -> str:
    if share_drifted >= config.DRIFT_ALERT_SHARE:
        return "alert"
    if share_drifted >= config.DRIFT_WARNING_SHARE or prediction_drifted:
        return "warning"
    return "ok"


def detect(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Feature and prediction drift of ``current`` against ``reference``.

    Both frames carry the raw features plus the ``churn_probability`` column.
    """
    if len(current) < config.DRIFT_MIN_ROWS:
        raise ValueError(
            f"Need at least {config.DRIFT_MIN_ROWS} rows to test for drift, got {len(current)}"
        )

    results = {
        **{c: _numeric_test(reference[c], current[c]) for c in NUMERIC_FIELDS},
        **{c: _categorical_test(reference[c], current[c]) for c in CATEGORICAL_FIELDS},
    }
    drifted = sorted(
        (
            {"feature": c, **{k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}}
            for c, r in results.items()
            if r["p_value"] < config.DRIFT_P_VALUE
        ),
        key=lambda d: d["p_value"],
    )
    share = len(drifted) / len(results)

    prediction = _numeric_test(reference[PREDICTION], current[PREDICTION])
    prediction_drifted = prediction["p_value"] < config.DRIFT_P_VALUE

    return {
        "status": status_for(share, prediction_drifted),
        "n_reference": len(reference),
        "n_current": len(current),
        "features_tested": len(results),
        "share_drifted": round(share, 4),
        "drifted_features": drifted,
        "p_values": {
            **{c: round(r["p_value"], 6) for c, r in results.items()},
            PREDICTION: round(prediction["p_value"], 6),
        },
        "prediction_drift": {
            "drifted": prediction_drifted,
            "statistic": round(prediction["statistic"], 4),
            "p_value": round(prediction["p_value"], 6),
            "reference_mean": round(float(reference[PREDICTION].mean()), 4),
            "current_mean": round(float(current[PREDICTION].mean()), 4),
        },
        "data_quality": data_quality(current),
        "thresholds": {
            "p_value": config.DRIFT_P_VALUE,
            "warning_share": config.DRIFT_WARNING_SHARE,
            "alert_share": config.DRIFT_ALERT_SHARE,
        },
    }


def score_frame(artifact: dict, records: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``records`` to the raw schema and attach the model's predictions."""
    current = prepare(records)
    current[PREDICTION] = artifact["pipeline"].predict_proba(current)[:, 1]
    return current


def analyse(artifact: dict, records: pd.DataFrame) -> dict:
    """Score ``records`` with the artifact's model and compare to its reference sample."""
    reference = artifact.get("drift_reference")
    if reference is None:
        raise LookupError("The model artifact has no drift reference; retrain to create one.")
    return detect(reference, score_frame(artifact, records))


# --------------------------------------------------------------------------- #
# Demo and HTML report
# --------------------------------------------------------------------------- #
def simulate_shift(frame: pd.DataFrame, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """A plausible bad month: newer, pricier, fiber-heavy customers and sloppy data."""
    rng = np.random.default_rng(seed)
    shifted = frame[config.FEATURES].copy().reset_index(drop=True)
    n = len(shifted)

    shifted["tenure"] = (shifted["tenure"] * 0.4).round().astype(int)
    shifted["MonthlyCharges"] = (shifted["MonthlyCharges"] * 1.25).round(2)
    shifted["TotalCharges"] = (shifted["TotalCharges"] * 0.4).round(2)

    shifted.loc[rng.random(n) < 0.6, "InternetService"] = "Fiber optic"
    shifted.loc[rng.random(n) < 0.7, "Contract"] = "Month-to-month"
    shifted.loc[rng.random(n) < 0.15, "PaymentMethod"] = "Crypto"  # never seen in training
    shifted.loc[rng.random(n) < 0.05, "MonthlyCharges"] = np.nan
    return shifted


def write_html_report(reference: pd.DataFrame, current: pd.DataFrame, path=config.DRIFT_REPORT_PATH):
    """Render Evidently's data-drift report, using the same tests and p-value as ``detect``.

    Needs the optional ``monitor`` extra (Evidently is large: ~400 MB installed).
    """
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        raise RuntimeError(
            "The HTML report needs Evidently: pip install -c constraints.txt '.[monitor]'"
        ) from exc

    definition = DataDefinition(
        numerical_columns=[*NUMERIC_FIELDS, PREDICTION],
        categorical_columns=CATEGORICAL_FIELDS,
    )
    as_dataset = lambda frame: Dataset.from_pandas(frame, data_definition=definition)
    report = Report(
        [
            DataDriftPreset(
                num_method="ks",
                cat_method="chisquare",
                num_threshold=config.DRIFT_P_VALUE,
                cat_threshold=config.DRIFT_P_VALUE,
                drift_share=config.DRIFT_ALERT_SHARE,
            )
        ]
    )
    snapshot = report.run(as_dataset(current), as_dataset(reference))
    config.ensure_dirs()
    snapshot.save_html(str(path))
    return path


def plot_summary(summary: dict, path=None) -> str:
    """Bar chart of drift evidence per field for the demo's clean and shifted batches."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = path or config.FIGURE_DIR / "drift_demo.png"
    shifted, clean = summary["shifted_batch"]["p_values"], summary["unshifted_batch"]["p_values"]
    names = sorted(shifted, key=lambda n: shifted[n])  # most drifted first
    evidence = lambda p: min(-np.log10(max(p, 1e-300)), 30)

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(names))
    ax.barh(y, [evidence(shifted[n]) for n in names], color="#dc2626", label="shifted batch")
    ax.scatter([evidence(clean[n]) for n in names], y, color="#2563eb", zorder=3, label="clean batch")
    ax.axvline(-np.log10(config.DRIFT_P_VALUE), color="#334155", linestyle="--", label="drift threshold")
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Evidence of drift: -log10(p-value), capped at 30")
    ax.set_title(
        f"Drift demo: shifted batch is {summary['shifted_batch']['status'].upper()}, "
        f"clean batch is {summary['unshifted_batch']['status'].upper()}"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def run_demo() -> dict:
    """Compare a clean batch and a simulated shifted batch against the reference."""
    import joblib

    from churnguard import data

    artifact = joblib.load(config.MODEL_PATH)
    reference = artifact["drift_reference"]
    _, _, X_test, *_ = data.split(data.load_clean())
    clean = X_test.sample(500, random_state=config.RANDOM_STATE)
    shifted = simulate_shift(clean)

    summary = {
        "unshifted_batch": analyse(artifact, clean),
        "shifted_batch": analyse(artifact, shifted),
    }
    config.ensure_dirs()
    config.DRIFT_DEMO_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for name, result in summary.items():
        print(
            f"{name}: {result['status'].upper()} - {len(result['drifted_features'])} of "
            f"{result['features_tested']} features drifted, prediction drift "
            f"{result['prediction_drift']['drifted']}, unknown-category rate "
            f"{result['data_quality']['overall_unknown_category_rate']:.1%}"
        )
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Summary chart: {plot_summary(summary)}")
    try:
        path = write_html_report(reference, score_frame(artifact, shifted))
        print(f"HTML report: {path}")
    except RuntimeError as exc:
        print(f"(no HTML report) {exc}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ChurnGuard drift monitoring.")
    parser.add_argument("--demo", action="store_true", help="simulate a shifted batch")
    args = parser.parse_args()
    if not args.demo:
        parser.error("nothing to do; pass --demo")
    run_demo()


if __name__ == "__main__":
    main()
