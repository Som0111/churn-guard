"""Central configuration: paths, schema and the business cost model.

Everything that a reviewer might want to tweak lives here rather than being
scattered through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RAW_CSV = RAW_DIR / "telco_churn.csv"
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_card.json"
METRICS_PATH = REPORT_DIR / "metrics.json"
CALIBRATION_PATH = REPORT_DIR / "calibration_comparison.json"

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
# SHA-256 of the raw CSV (970,457 bytes). A mismatch means the upstream file
# changed or the cached copy was altered, so every number downstream would drift.
DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
VAL_SIZE = 0.2   # threshold tuning only
TEST_SIZE = 0.2  # final report only, scored once
CV_FOLDS = 5
CALIBRATION_TIE = 0.001  # Brier+ECE gap below which the simpler variant wins


# --------------------------------------------------------------------------- #
# Business cost model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostModel:
    """Turns a confusion matrix into money.

    The retention team sends a discount offer to every customer the model
    flags. Relative to the do-nothing baseline:

    * True positive  - we pay for the offer and save the customer with
      probability ``offer_success_rate``.
    * False positive - we pay for an offer a loyal customer did not need.
    * True negative / false negative - no action, no cost, no gain. These are
      the baseline, which is why they are worth exactly zero here.
    """

    offer_cost: float = 50.0            # incentive handed to a flagged customer
    customer_lifetime_value: float = 500.0   # margin lost when a customer leaves
    offer_success_rate: float = 0.30    # share of true churners the offer saves

    @property
    def true_positive_value(self) -> float:
        return self.offer_success_rate * self.customer_lifetime_value - self.offer_cost

    @property
    def false_positive_value(self) -> float:
        return -self.offer_cost

    def net_benefit(self, tp: int, fp: int) -> float:
        """Campaign profit in dollars for a given confusion matrix."""
        return tp * self.true_positive_value + fp * self.false_positive_value


DEFAULT_COST_MODEL = CostModel()


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for path in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)
