"""Evaluation, threshold optimisation and report figures.

The central idea of this project: ranking quality (ROC-AUC) is what the model
controls, but the retention team acts on a *decision*, and the decision has a
price. So the threshold is chosen by maximising campaign profit under
``config.CostModel`` rather than by defaulting to 0.5.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churnguard import config
from churnguard.config import DEFAULT_COST_MODEL, CostModel

logger = logging.getLogger(__name__)


def _repo_relative(path) -> str:
    """Portable path for the report, so metrics.json is not machine-specific."""
    from pathlib import Path

    path = Path(path).resolve()
    try:
        return path.relative_to(config.ROOT).as_posix()
    except ValueError:  # outside the repo (e.g. a temp dir in tests)
        return path.as_posix()


def _calibration_bins(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int
) -> list[tuple[int, float, float]]:
    """Equal-count bins of (size, mean predicted, observed rate)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    order = np.argsort(proba, kind="stable")
    return [
        (len(idx), float(proba[idx].mean()), float(y_true[idx].mean()))
        for idx in np.array_split(order, n_bins)
        if len(idx)
    ]


def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    """Size-weighted mean |predicted - observed| over equal-count bins."""
    bins = _calibration_bins(y_true, proba, n_bins)
    total = sum(n for n, _, _ in bins)
    return sum(n * abs(pred - obs) for n, pred, obs in bins) / total


def reliability_table(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> list[dict]:
    """Mean predicted vs observed churn rate per bin, lowest scores first."""
    return [
        {
            "bin": i + 1,
            "n": n,
            "mean_predicted": round(pred, 4),
            "observed_rate": round(obs, 4),
        }
        for i, (n, pred, obs) in enumerate(_calibration_bins(y_true, proba, n_bins))
    ]


def classification_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> dict:
    """Threshold-free ranking metrics plus metrics at a chosen threshold."""
    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "classification_metrics needs both classes in y_true; ROC-AUC and PR-AUC "
            "are undefined for a single-class sample"
        )
    y_pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "brier_score": round(float(brier_score_loss(y_true, proba)), 4),
        "ece": round(expected_calibration_error(y_true, proba), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def profit_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    cost_model: CostModel = DEFAULT_COST_MODEL,
) -> pd.DataFrame:
    """Campaign profit at every threshold that changes a decision.

    Candidates are the unique predicted probabilities plus 0 and 1, so the true
    optimum is always on the list - there is no fixed grid to fall between.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    positives = int(np.sum(y_true == 1))
    rows = []

    for threshold in np.unique(np.concatenate([[0.0, 1.0], proba])):
        flagged = proba >= threshold
        tp = int(np.sum(flagged & (y_true == 1)))
        fp = int(np.sum(flagged & (y_true == 0)))
        rows.append(
            {
                "threshold": threshold,
                "customers_targeted": tp + fp,
                "churners_caught": tp,
                "wasted_offers": fp,
                "churners_missed": positives - tp,
                "net_benefit": cost_model.net_benefit(tp, fp),
            }
        )

    return pd.DataFrame(rows)


def apply_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    cost_model: CostModel = DEFAULT_COST_MODEL,
) -> dict:
    """Business outcome of a *fixed* threshold on this data - nothing is tuned.

    Also reports the naive 0.5 threshold and the no-model baselines so the gain
    is explicit rather than implied. The threshold is kept unrounded: rounding
    it could flip decisions on the very scores it was chosen from.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    def outcome(t: float) -> tuple[int, int]:
        flagged = proba >= t
        return int(np.sum(flagged & (y_true == 1))), int(np.sum(flagged & (y_true == 0)))

    tp, fp = outcome(threshold)
    benefit = cost_model.net_benefit(tp, fp)
    default_benefit = cost_model.net_benefit(*outcome(0.5))

    # The two baselines a retention team could run without any model at all.
    n = len(y_true)
    n_churners = int(np.sum(y_true))
    blanket_benefit = cost_model.net_benefit(n_churners, n - n_churners)

    return {
        "optimal_threshold": float(threshold),
        "net_benefit_optimal": round(float(benefit), 2),
        "net_benefit_at_0.5": round(float(default_benefit), 2),
        "net_benefit_blanket_campaign": round(float(blanket_benefit), 2),
        "net_benefit_do_nothing": 0.0,
        "uplift_vs_default": round(float(benefit - default_benefit), 2),
        "uplift_vs_blanket": round(float(benefit - blanket_benefit), 2),
        "benefit_per_customer": round(float(benefit) / n, 2),
        "customers_targeted": tp + fp,
        "churners_caught": tp,
        "wasted_offers": fp,
        "churners_missed": n_churners - tp,
        "assumptions": {
            "offer_cost": cost_model.offer_cost,
            "customer_lifetime_value": cost_model.customer_lifetime_value,
            "offer_success_rate": cost_model.offer_success_rate,
            "value_per_true_positive": cost_model.true_positive_value,
        },
    }


def optimize_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    cost_model: CostModel = DEFAULT_COST_MODEL,
) -> dict:
    """Pick the profit-maximising threshold on this data. Validation data only."""
    curve = profit_curve(y_true, proba, cost_model)
    best = float(curve.loc[curve["net_benefit"].idxmax(), "threshold"])
    return apply_threshold(y_true, proba, best, cost_model)


# --------------------------------------------------------------------------- #
# Uncertainty and sensitivity
# --------------------------------------------------------------------------- #
def bootstrap_intervals(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    cost_model: CostModel = DEFAULT_COST_MODEL,
    n_resamples: int = config.BOOTSTRAP_RESAMPLES,
    level: float = config.BOOTSTRAP_LEVEL,
    seed: int = config.RANDOM_STATE,
) -> dict:
    """Percentile bootstrap of the test set at a *frozen* threshold.

    Captures sampling noise in the test customers only. It does not capture
    uncertainty in the threshold choice, the cost assumptions, or the model fit.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    def stats(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
        flagged = p >= threshold
        tp = int(np.sum(flagged & (y == 1)))
        fp = int(np.sum(flagged & (y == 0)))
        positives = int(np.sum(y == 1))
        return {
            "profit_model": cost_model.net_benefit(tp, fp),
            "profit_blanket": cost_model.net_benefit(positives, n - positives),
            "profit_do_nothing": 0.0,
            "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / positives if positives else 0.0,
        }

    point = stats(y_true, proba)
    draws = {k: [] for k in point}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:  # vanishingly rare; AUC undefined
            continue
        for k, v in stats(y_true[idx], proba[idx]).items():
            draws[k].append(v)

    low, high = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return {
        "n_resamples": n_resamples,
        "seed": seed,
        "level": level,
        "threshold": float(threshold),
        "scope": "test-set sampling noise only, at the frozen threshold",
        **{
            k: {
                "point": round(float(point[k]), 4),
                "low": round(float(np.percentile(draws[k], low)), 4),
                "high": round(float(np.percentile(draws[k], high)), 4),
            }
            for k in point
        },
    }


def sensitivity_grid(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    base: CostModel = DEFAULT_COST_MODEL,
    rates: tuple[float, ...] = tuple(round(0.10 + 0.05 * i, 2) for i in range(9)),
    costs: tuple[float, ...] = tuple(25.0 + 5.0 * i for i in range(16)),
) -> dict:
    """Test-set profit of the shipped threshold across offer success rate x offer cost.

    The threshold is held fixed (it was tuned for the base assumptions), so this
    shows how the *deployed* policy fares if the assumptions are wrong - not what
    a re-tuned policy would earn.
    """
    y_true = np.asarray(y_true)
    flagged = np.asarray(proba) >= threshold
    tp = int(np.sum(flagged & (y_true == 1)))
    fp = int(np.sum(flagged & (y_true == 0)))
    positives, n = int(np.sum(y_true == 1)), len(y_true)

    model, blanket = [], []
    for cost in costs:
        row_m, row_b = [], []
        for rate in rates:
            cm = CostModel(cost, base.customer_lifetime_value, rate)
            row_m.append(round(cm.net_benefit(tp, fp), 2))
            row_b.append(round(cm.net_benefit(positives, n - positives), 2))
        model.append(row_m)
        blanket.append(row_b)

    m, b = np.array(model), np.array(blanket)
    advantage = m - b

    def first_rate(mask_row: np.ndarray) -> float | None:
        hits = np.flatnonzero(mask_row)
        return float(rates[hits[0]]) if len(hits) else None

    return {
        "threshold": float(threshold),
        "customer_lifetime_value": base.customer_lifetime_value,
        "offer_success_rates": list(rates),
        "offer_costs": list(costs),
        "model_profit": model,  # rows = offer cost, columns = success rate
        "blanket_profit": blanket,
        "model_minus_blanket": advantage.round(2).tolist(),
        "cells_where_blanket_wins": int(np.sum(advantage <= 0)),
        "cells_where_model_loses_money": int(np.sum(m < 0)),
        "cells_total": int(m.size),
        # per offer cost: lowest grid success rate at which the model beats blanket / is profitable
        "min_success_rate_model_profitable": {
            str(c): first_rate(m[i] > 0) for i, c in enumerate(costs)
        },
        "max_success_rate_model_beats_blanket": {
            str(c): (
                float(rates[np.flatnonzero(advantage[i] > 0)[-1]])
                if np.any(advantage[i] > 0)
                else None
            )
            for i, c in enumerate(costs)
        },
    }


def plot_sensitivity(grid: dict) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.ensure_dirs()
    rates, costs = grid["offer_success_rates"], grid["offer_costs"]
    panels = [
        ("Model campaign profit ($)", np.array(grid["model_profit"]), "RdBu"),
        ("Model minus blanket campaign ($)", np.array(grid["model_minus_blanket"]), "RdBu"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (title, values, cmap) in zip(axes, panels, strict=True):
        limit = float(np.abs(values).max())
        im = ax.imshow(
            values, origin="lower", aspect="auto", cmap=cmap, vmin=-limit, vmax=limit
        )
        ax.contour(values, levels=[0], colors="black", linewidths=1.2)
        ax.set_xticks(range(len(rates)), [f"{r:.0%}" for r in rates])
        ax.set_yticks(range(0, len(costs), 3), [f"${costs[i]:.0f}" for i in range(0, len(costs), 3)])
        ax.set_xlabel("Offer success rate (assumed)")
        ax.set_ylabel("Offer cost (assumed)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        base_x, base_y = rates.index(0.3), costs.index(50.0)
        ax.plot(base_x, base_y, marker="o", color="black", markersize=7, fillstyle="none")
    fig.suptitle("Sensitivity at the shipped threshold (black ring = $50 / 30% assumption)")
    fig.tight_layout()
    path = config.FIGURE_DIR / "sensitivity_heatmap.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return _repo_relative(path)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def generate_figures(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    cost_model: CostModel = DEFAULT_COST_MODEL,
    variants: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[str]:
    """Write the four report figures and return their paths.

    ``variants`` maps a calibration variant name to its (y, proba) on validation.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve

    config.ensure_dirs()
    written: list[str] = []
    accent, muted = "#2563eb", "#94a3b8"

    # 1. ROC + precision-recall side by side. Drawn from the raw curve arrays
    # rather than the Display helpers, whose kwargs shift between sklearn
    # releases.
    fpr, tpr, _ = roc_curve(y_true, proba)
    precision, recall, _ = precision_recall_curve(y_true, proba)
    prevalence = float(np.mean(y_true))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(
        fpr, tpr, color=accent, linewidth=2, label=f"AUC = {roc_auc_score(y_true, proba):.3f}"
    )
    axes[0].plot([0, 1], [0, 1], "--", color=muted, linewidth=1, label="random")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend(loc="lower right")

    axes[1].plot(
        recall,
        precision,
        color=accent,
        linewidth=2,
        label=f"AP = {average_precision_score(y_true, proba):.3f}",
    )
    axes[1].axhline(
        prevalence, linestyle="--", color=muted, linewidth=1, label=f"base rate = {prevalence:.2f}"
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    path = config.FIGURE_DIR / "roc_pr_curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(_repo_relative(path))

    # 2. Profit curve - the headline chart
    curve = profit_curve(y_true, proba, cost_model)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve["threshold"], curve["net_benefit"], color=accent, linewidth=2)
    ax.axvline(
        threshold,
        linestyle="--",
        color="#dc2626",
        label=f"chosen on validation = {threshold:.2f}",
    )
    ax.axvline(0.5, linestyle=":", color=muted, label="default = 0.50")
    ax.axhline(0, color="#334155", linewidth=0.8, label="do nothing")

    n_churners = int(np.sum(y_true))
    blanket = cost_model.net_benefit(n_churners, len(y_true) - n_churners)
    ax.axhline(
        blanket,
        linestyle="--",
        color="#ea580c",
        linewidth=1.2,
        label=f"blanket campaign = ${blanket:,.0f}",
    )

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Campaign net benefit ($, test set)")
    ax.set_title("Targeted retention turns a loss-making campaign profitable")
    ax.legend(loc="lower center", fontsize=9)
    fig.tight_layout()
    path = config.FIGURE_DIR / "profit_curve.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(_repo_relative(path))

    # 3. Confusion matrix at the chosen threshold
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            f"{value:,}",
            ha="center",
            va="center",
            color="white" if value > cm.max() / 2 else "#0f172a",
            fontsize=13,
        )
    ax.set_xticks([0, 1], ["stayed", "churned"])
    ax.set_yticks([0, 1], ["stayed", "churned"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix @ threshold {threshold:.2f}")
    fig.tight_layout()
    path = config.FIGURE_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(_repo_relative(path))

    # 4. Calibration - are the probabilities trustworthy enough to price?
    # Left: the four variants compared on validation. Right: the chosen model on test.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax in axes:
        ax.plot([0, 1], [0, 1], "--", color=muted, label="perfect calibration")
        ax.set_xlabel("Predicted churn probability")
        ax.set_ylabel("Observed churn rate")

    for name, (y_v, p_v) in (variants or {}).items():
        v_true, v_pred = calibration_curve(y_v, p_v, n_bins=10, strategy="quantile")
        axes[0].plot(
            v_pred, v_true, "o-", label=f"{name} (ECE {expected_calibration_error(y_v, p_v):.3f})"
        )
    axes[0].set_title("Calibration variants (validation)")
    axes[0].legend(fontsize=8)

    prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    axes[1].plot(
        prob_pred,
        prob_true,
        "o-",
        color=accent,
        label=f"ChurnGuard (ECE {expected_calibration_error(y_true, proba):.3f})",
    )
    axes[1].set_title("Chosen model (test)")
    axes[1].legend()
    fig.tight_layout()
    path = config.FIGURE_DIR / "calibration.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(_repo_relative(path))

    logger.info("Wrote %d figures to %s", len(written), config.FIGURE_DIR)
    return written


def save_metrics(payload: dict) -> None:
    config.ensure_dirs()
    config.METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Metrics written to %s", config.METRICS_PATH)
