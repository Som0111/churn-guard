"""Time-to-churn analysis: *when* customers leave, not just *whether* they look like leavers.

The classifier in the rest of this project answers "does this customer resemble a
churner?". Survival analysis uses tenure as the clock and answers "how does the
churn hazard - the rate of leaving among customers still active - change over a
customer's life, and for whom?". Analysis and reports only: no API endpoint.

Needs the optional extra:  pip install -c constraints.txt ".[analysis]"
Run:                       python -m churnguard.survival
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from churnguard import config, data, evaluate, features

logger = logging.getLogger(__name__)

HORIZONS = (12, 24, 36, 60)  # months
KM_GROUPS = ("Contract", "InternetService")
COX_NUMERIC = ["MonthlyCharges", "SeniorCitizen", "n_addon_services"]
COX_CATEGORICAL = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "PaperlessBilling",
    "Partner",
    "Dependents",
]
COX_PENALIZER = 0.01  # small ridge: monthly charge and internet type are strongly correlated
PH_ALPHA = 0.01       # a covariate "violates" proportional hazards below this p-value
SUMMARY_JSON = config.REPORT_DIR / "survival_summary.json"
SUMMARY_MD = config.REPORT_DIR / "survival_summary.md"


def _lifelines():
    try:
        import lifelines
    except ImportError as exc:
        raise RuntimeError(
            "Survival analysis needs lifelines: pip install -c constraints.txt '.[analysis]'"
        ) from exc
    return lifelines


def _p(value: float) -> str:
    return "< 1e-100" if value < 1e-100 else f"= {value:.2g}"


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop customers with no observed time at risk (tenure 0); return (frame, n_dropped)."""
    kept = df[df["tenure"] > 0].copy()
    return kept, len(df) - len(kept)


def kaplan_meier(df: pd.DataFrame, column: str) -> dict:
    """Survival curve per level of ``column`` plus a log-rank test across levels."""
    lifelines = _lifelines()
    from lifelines.statistics import multivariate_logrank_test

    groups = {}
    for level, part in df.groupby(column):
        fit = lifelines.KaplanMeierFitter().fit(part["tenure"], part[config.TARGET], label=str(level))
        survival = fit.survival_function_at_times(list(HORIZONS)).to_numpy()
        median = fit.median_survival_time_
        groups[str(level)] = {
            "n": len(part),
            "churned": int(part[config.TARGET].sum()),
            "survival_at_months": {str(t): round(float(s), 4) for t, s in zip(HORIZONS, survival, strict=True)},
            "median_months": None if np.isinf(median) else float(median),
            "_fit": fit,
        }
    test = multivariate_logrank_test(df["tenure"], df[column], df[config.TARGET])
    return {"column": column, "groups": groups, "logrank_p_value": float(test.p_value)}


def cox_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Covariates + duration + event. TotalCharges is excluded on purpose: it is
    roughly tenure x monthly charge, so it would leak the clock into the covariates."""
    enriched = features.add_domain_features(df)
    frame = enriched[COX_NUMERIC + COX_CATEGORICAL].copy()
    frame["MonthlyCharges"] = frame["MonthlyCharges"] / 10.0  # hazard ratio per $10
    frame = frame.rename(columns={"MonthlyCharges": "MonthlyCharges_per_10usd"})
    frame = pd.get_dummies(frame, columns=COX_CATEGORICAL, drop_first=True, dtype=float)
    frame["tenure"] = enriched["tenure"].to_numpy()
    frame[config.TARGET] = enriched[config.TARGET].to_numpy()
    return frame


def fit_cox(df: pd.DataFrame, train_index, val_index) -> dict:
    """Cox proportional-hazards model fit on train, concordance reported on validation."""
    lifelines = _lifelines()
    from lifelines.statistics import proportional_hazard_test

    frame = cox_frame(df)  # dummies built once so train and validation share columns
    train, val = frame.loc[train_index], frame.loc[val_index]

    cph = lifelines.CoxPHFitter(penalizer=COX_PENALIZER)
    cph.fit(train, duration_col="tenure", event_col=config.TARGET)

    ph = proportional_hazard_test(cph, train, time_transform="rank").summary
    hazard = cph.summary
    return {
        "n_train": len(train),
        "n_validation": len(val),
        "penalizer": COX_PENALIZER,
        "concordance_train": round(float(cph.concordance_index_), 4),
        "concordance_validation": round(
            float(cph.score(val, scoring_method="concordance_index")), 4
        ),
        "hazard_ratios": {
            name: {
                "hazard_ratio": round(float(row["exp(coef)"]), 4),
                "ci_low": round(float(row["exp(coef) lower 95%"]), 4),
                "ci_high": round(float(row["exp(coef) upper 95%"]), 4),
                "p_value": float(row["p"]),
            }
            for name, row in hazard.iterrows()
        },
        "proportional_hazards": {
            "alpha": PH_ALPHA,
            "p_values": {name: float(p) for name, p in ph["p"].items()},
            "violations": sorted(name for name, p in ph["p"].items() if p < PH_ALPHA),
        },
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_km(results: list[dict]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.5), sharey=True)
    for ax, result in zip(np.atleast_1d(axes), results, strict=True):
        for group in result["groups"].values():
            group["_fit"].plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.12)
        ax.set_title(f"Survival by {result['column']} (log-rank p {_p(result['logrank_p_value'])})")
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Share of customers still active")
        ax.set_ylim(0, 1.02)
    fig.tight_layout()
    path = config.FIGURE_DIR / "survival_km.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return evaluate._repo_relative(path)


def plot_hazard_ratios(cox: dict) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(cox["hazard_ratios"], key=lambda n: cox["hazard_ratios"][n]["hazard_ratio"])
    hr = np.array([cox["hazard_ratios"][n]["hazard_ratio"] for n in names])
    low = np.array([cox["hazard_ratios"][n]["ci_low"] for n in names])
    high = np.array([cox["hazard_ratios"][n]["ci_high"] for n in names])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(hr, range(len(names)), xerr=[hr - low, high - hr], fmt="o", color="#2563eb", capsize=3)
    ax.axvline(1.0, color="#334155", linestyle="--")
    ax.set_xscale("log")
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("Hazard ratio (log scale; above 1 = leaves sooner)")
    ax.set_title(f"Cox model - validation concordance {cox['concordance_validation']:.3f}")
    fig.tight_layout()
    path = config.FIGURE_DIR / "survival_cox_hazard_ratios.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return evaluate._repo_relative(path)


# --------------------------------------------------------------------------- #
# Orchestration and report
# --------------------------------------------------------------------------- #
def run(df: pd.DataFrame | None = None, write_figures: bool = True) -> dict:
    """KM curves + Cox model on the cleaned data, using the project's train/validation split."""
    df = data.load_clean() if df is None else df
    X_train, X_val, _X_test, *_ = data.split(df)  # the test split stays untouched
    df, dropped = prepare(df)

    km = [kaplan_meier(df, column) for column in KM_GROUPS]
    cox = fit_cox(
        df,
        [i for i in X_train.index if i in df.index],
        [i for i in X_val.index if i in df.index],
    )

    figures = []
    if write_figures:
        config.ensure_dirs()
        figures = [plot_km(km), plot_hazard_ratios(cox)]

    return {
        "n_customers": len(df),
        "n_dropped_zero_tenure": dropped,
        "kaplan_meier": [
            {**k, "groups": {g: {a: b for a, b in v.items() if a != "_fit"} for g, v in k["groups"].items()}}
            for k in km
        ],
        "cox": cox,
        "figures": figures,
    }


def render_markdown(result: dict) -> str:
    cox = result["cox"]
    ph = cox["proportional_hazards"]
    lines = [
        "# Time to churn - survival analysis",
        "",
(        f"{result['n_customers']:,} customers; {result['n_dropped_zero_tenure']} with tenure 0 dropped "
            "(no observed time at risk). Clock = tenure in months; event = churned. "
            "Customers who had not churned are *censored*: we know they lasted at least their tenure, no more."),
        "",
        "## Hazard vs probability",
        "",
(        "The classifier gives a **probability**: how likely this customer is flagged as churned in the "
            "snapshot. Survival analysis models the **hazard**: the rate of leaving *at a given tenure, among "
            "customers still active then*. A hazard ratio of 2 means \"leaves at twice the rate at any given "
            "tenure\", not \"twice the probability of ever leaving\". Hazards are what let us say *when*."),
        "",
        "## Kaplan-Meier: share still active",
        "",
    ]
    for km in result["kaplan_meier"]:
        header = " | ".join(f"{t} mo" for t in HORIZONS)
        lines += [
            f"**By {km['column']}** (log-rank p {_p(km['logrank_p_value'])})",
            "",
            f"| {km['column']} | customers | churned | {header} | median |",
            "|---|---|---|" + "---|" * (len(HORIZONS) + 1),
        ]
        for name, g in km["groups"].items():
            cells = " | ".join(f"{g['survival_at_months'][str(t)]:.1%}" for t in HORIZONS)
            median = "not reached" if g["median_months"] is None else f"{g['median_months']:.0f} mo"
            lines.append(f"| {name} | {g['n']:,} | {g['churned']:,} | {cells} | {median} |")
        lines.append("")

    lines += [
        "![Kaplan-Meier curves](figures/survival_km.png)",
        "",
        "## Cox proportional-hazards model",
        "",
(        f"Fit on the training split ({cox['n_train']:,} customers), scored on validation "
            f"({cox['n_validation']:,}). **Concordance: {cox['concordance_validation']:.3f} on validation** "
            f"({cox['concordance_train']:.3f} on train; 0.5 is chance, 1.0 is perfect ranking of who leaves "
            f"first). Ridge penalty {cox['penalizer']}. `TotalCharges` is excluded because it is roughly "
            "tenure x monthly charge and would leak the clock into the covariates."),
        "",
        "| Covariate | Hazard ratio | 95% CI | p |",
        "|---|---|---|---|",
    ]
    for name, h in sorted(cox["hazard_ratios"].items(), key=lambda kv: -kv[1]["hazard_ratio"]):
        lines.append(
            f"| {name} | {h['hazard_ratio']:.2f} | {h['ci_low']:.2f} to {h['ci_high']:.2f} | {h['p_value']:.2g} |"
        )
    lines += [
        "",
        "![Hazard ratios](figures/survival_cox_hazard_ratios.png)",
        "",
        "## Is proportional hazards credible?",
        "",
        f"Schoenfeld-residual test (rank time transform), flagging p < {ph['alpha']}:",
        "",
    ]
    if ph["violations"]:
        lines.append(
            f"**{len(ph['violations'])} of {len(ph['p_values'])} covariates violate the assumption:** "
            + ", ".join(f"`{n}` (p = {ph['p_values'][n]:.2g})" for n in ph["violations"])
            + ". For these, one hazard ratio is an average over a changing effect - the effect is "
            "not constant across tenure - so read them as a summary, not as a fixed multiplier. "
            "The Kaplan-Meier curves above do not assume proportionality and are the safer evidence "
            "for the contract and internet-service comparisons."
        )
    else:
        lines.append("No covariate is flagged.")
    lines += [
        "",
(        "With several thousand customers this test flags even small departures, so a flag means "
            "\"the effect varies with tenure\", not necessarily \"the model is unusable\"."),
        "",
        "*Associations in observational data, not causal effects.*",
        "",
    ]
    return "\n".join(lines)


def main() -> dict:
    result = run()
    SUMMARY_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    SUMMARY_MD.write_text(render_markdown(result), encoding="utf-8")
    cox = result["cox"]
    logger.info(
        "Cox validation concordance %.3f; PH violations: %s",
        cox["concordance_validation"],
        cox["proportional_hazards"]["violations"],
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    main()
