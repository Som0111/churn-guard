"""Why did permutation importance say shuffling `tenure` costs 0.37 ROC-AUC?

A model whose AUC is 0.846 cannot lose more than 0.346 and still beat a coin flip,
so a drop of 0.37 means the shuffled rows were not just uninformative - they were
broken. This script reproduces the number and shows the mechanism.

Run (after training):  python scripts/investigate_permutation_importance.py
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from churnguard import config, data, features
from churnguard.train import permutation_drop


def main() -> None:
    model = joblib.load(config.MODEL_PATH)["pipeline"]
    _, _, X, _, _, y = data.split(data.load_clean())
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print(f"Test ROC-AUC: {base:.4f}\n")

    print("1. Raw-field shuffles (drop in AUC, mean +/- std over 20 shuffles)")
    for columns in (
        ["tenure"],
        ["TotalCharges"],
        ["MonthlyCharges"],
        ["Contract"],
        ["tenure", "TotalCharges"],
        ["tenure", "TotalCharges", "MonthlyCharges"],
    ):
        mean, std = permutation_drop(model, X, y, columns, n_repeats=20)
        print(f"   {' + '.join(columns):42s} {mean:.4f} +/- {std:.4f}")

    print("\n2. What shuffling `tenure` alone does to the engineered features")
    rng = np.random.default_rng(config.RANDOM_STATE)
    shuffled = X.copy()
    shuffled["tenure"] = X["tenure"].to_numpy()[rng.permutation(len(X))]
    real, broken = features.add_domain_features(X), features.add_domain_features(shuffled)
    for column in ("avg_monthly_spend", "spend_vs_current_ratio"):
        print(
            f"   {column:24s} 99th pct real {real[column].quantile(0.99):8.1f} "
            f"-> after shuffle {broken[column].quantile(0.99):8.1f}"
        )
    auc = roc_auc_score(y, model.predict_proba(shuffled)[:, 1])
    print(f"   AUC after the tenure-only shuffle: {auc:.4f}  (a coin flip is 0.5)")

    print("\n3. Tenure-derived features, shuffled in the model's own input space")
    stage, estimator = model.named_steps["features"], model.named_steps["model"]
    names, matrix = features.feature_names(model), stage.transform(X)

    def drop(columns: list[str], repeats: int = 20) -> float:
        idx = [names.index(c) for c in columns]
        drops = []
        for _ in range(repeats):
            shuffled_matrix = matrix.copy()
            shuffled_matrix[:, idx] = matrix[rng.permutation(len(matrix))][:, idx]
            drops.append(base - roc_auc_score(y, estimator.predict_proba(shuffled_matrix)[:, 1]))
        return float(np.mean(drops))

    group = ["tenure", "tenure_years", "is_new_customer", "avg_monthly_spend", "spend_vs_current_ratio"]
    for column in group:
        print(f"   {column:24s} alone  {drop([column]):.4f}")
    print(f"   {'all five, jointly':24s} block  {drop(group):.4f}")
    print(
        "\n   Individually each is small because they carry the same signal (redundancy);\n"
        "   shuffled together they show how much the model leans on tenure information."
    )


if __name__ == "__main__":
    main()
