"""FastAPI serving layer.

Loads the fitted pipeline once at startup and exposes scoring endpoints. The
request schema mirrors the raw customer record, so a caller never has to know
anything about the feature engineering that happens inside.

Run with:  uvicorn churnguard.api:app --reload
Docs at :  http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from churnguard import __version__, config, drift, explain

logger = logging.getLogger(__name__)

MODEL: dict = {}

Yes_No = Literal["Yes", "No"]


class Customer(BaseModel):
    """One raw customer record, exactly as it appears in the source system."""

    gender: Literal["Male", "Female"] = "Female"
    SeniorCitizen: int = Field(0, ge=0, le=1)
    Partner: Yes_No = "No"
    Dependents: Yes_No = "No"
    tenure: int = Field(..., ge=0, le=100, description="Months as a customer")
    PhoneService: Yes_No = "Yes"
    MultipleLines: Literal["Yes", "No", "No phone service"] = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = "No"
    OnlineBackup: Literal["Yes", "No", "No internet service"] = "No"
    DeviceProtection: Literal["Yes", "No", "No internet service"] = "No"
    TechSupport: Literal["Yes", "No", "No internet service"] = "No"
    StreamingTV: Literal["Yes", "No", "No internet service"] = "No"
    StreamingMovies: Literal["Yes", "No", "No internet service"] = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: Yes_No = "Yes"
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = "Electronic check"
    MonthlyCharges: float = Field(
        ..., ge=0, le=config.NUMERIC_RANGES["MonthlyCharges"][1], allow_inf_nan=False
    )
    TotalCharges: float = Field(
        ..., ge=0, le=config.NUMERIC_RANGES["TotalCharges"][1], allow_inf_nan=False
    )

    @model_validator(mode="after")
    def _zero_tenure_has_no_history(self) -> Customer:
        # Deliberately loose: a brand-new customer can carry a first partial bill,
        # but not more than one month's charge. Everything else is left to the model.
        if self.tenure == 0 and self.TotalCharges > self.MonthlyCharges:
            raise ValueError(
                "TotalCharges exceeds MonthlyCharges for a customer with tenure 0; "
                "a brand-new customer cannot have more than one month of billing"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "No",
                    "Dependents": "No",
                    "tenure": 2,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 79.85,
                    "TotalCharges": 159.7,
                }
            ]
        }
    }


class Driver(BaseModel):
    """One reason behind a score, in terms of an original customer field."""

    feature: str
    direction: Literal["raises", "lowers"]
    magnitude: float = Field(..., description="Absolute SHAP contribution, log-odds of churn")


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_band: Literal["low", "medium", "high"]
    threshold_used: float
    recommended_action: str
    top_drivers: list[Driver] | None = Field(
        None, description="Top 3 fields moving this score; null if SHAP is unavailable"
    )


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(..., min_length=1, max_length=1000)


class DriftRequest(BaseModel):
    """Raw customer records from a recent period.

    Deliberately NOT validated like ``Customer``: drift monitoring exists to see
    data the scoring schema would reject - new categories, missing or garbled values.
    """

    records: list[dict[str, Any]] = Field(
        ..., min_length=config.DRIFT_MIN_ROWS, max_length=10_000
    )


def load_model() -> dict:
    """Load the artifact written by ``churnguard.train``."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model at {config.MODEL_PATH}. Run `python -m churnguard.train` first."
        )
    return joblib.load(config.MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        MODEL.update(load_model())
        MODEL["explainer"] = _build_explainer(MODEL)
        logger.info(
            "Loaded %s (threshold %.2f)", MODEL["model_name"], MODEL["threshold"]
        )
    except Exception as exc:  # noqa: BLE001 - missing OR corrupt artifact must not crash startup
        # Start anyway so /health can report the problem instead of crash-looping.
        logger.error("Could not load model: %s: %s", type(exc).__name__, exc)
    yield
    MODEL.clear()


app = FastAPI(
    title="ChurnGuard API",
    description=(
        "Cost-sensitive churn scoring. The decision threshold is not 0.5 - it is "
        "the value that maximises retention-campaign profit on held-out data."
    ),
    version=__version__,
    lifespan=lifespan,
)


ERROR_CODES = {
    404: "not_found",
    405: "method_not_allowed",
    422: "validation_error",
    500: "internal_error",
    503: "model_unavailable",
}


def _error(status: int, detail, code: str | None = None) -> JSONResponse:
    """Every error body is ``{"detail": ..., "error_code": ...}``."""
    return JSONResponse(
        status_code=status,
        content={"detail": detail, "error_code": code or ERROR_CODES.get(status, "error")},
    )


@app.exception_handler(StarletteHTTPException)
async def http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _error(exc.status_code, exc.detail)


@app.exception_handler(Exception)
async def unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
    # Class name only: exception messages can quote the customer data that caused them.
    logger.error("Unhandled %s", type(exc).__name__)
    return _error(500, "Internal server error.")


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """Readable 422s: field + message only.

    The default body echoes the rejected input, which crashes the response for
    NaN/Infinity (not valid JSON) and would also echo customer data back.
    """
    detail = [
        {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
    ]
    return _error(422, detail)


def _build_explainer(artifact: dict):
    try:
        return explain.build_explainer(artifact["pipeline"], artifact.get("explainer_background"))
    except Exception as exc:  # noqa: BLE001 - explanations are optional; scoring must still start
        logger.error("Could not build explainer: %s: %s", type(exc).__name__, exc)
        return None


def _drivers(frame: pd.DataFrame) -> list[list[dict] | None]:
    explainer = MODEL.get("explainer")
    if explainer is None:
        return [None] * len(frame)
    try:
        return explainer.drivers(frame)
    except Exception as exc:  # noqa: BLE001 - a failed explanation must not fail the score
        logger.error("Explanation failed: %s: %s", type(exc).__name__, exc)
        return [None] * len(frame)


def _risk_band(probability: float, threshold: float) -> str:
    if probability >= min(0.75, threshold + 0.25):
        return "high"
    if probability >= threshold:
        return "medium"
    return "low"


def _action(band: str) -> str:
    return {
        "high": "Priority outreach: call within 48h and offer a contract upgrade.",
        "medium": "Include in the next automated retention-offer batch.",
        "low": "No action - targeting this customer costs more than it returns.",
    }[band]


def _score(frame: pd.DataFrame) -> list[Prediction]:
    if not MODEL:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m churnguard.train`, then restart.",
        )

    try:
        probabilities = MODEL["pipeline"].predict_proba(frame)[:, 1]
    except Exception as exc:
        # Never log the payload or the exception message (it can quote customer values).
        logger.error("Inference failed (%s) on a batch of %d", type(exc).__name__, len(frame))
        raise HTTPException(status_code=500, detail="Scoring failed. The error has been logged.") from exc
    threshold = float(MODEL["threshold"])

    drivers = _drivers(frame)
    results = []
    for probability, reasons in zip(probabilities, drivers, strict=True):
        band = _risk_band(float(probability), threshold)
        results.append(
            Prediction(
                churn_probability=round(float(probability), 4),
                will_churn=bool(probability >= threshold),
                risk_band=band,
                threshold_used=round(threshold, 4),
                recommended_action=_action(band),
                top_drivers=reasons,
            )
        )
    return results


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send anyone who opens the bare domain to the interactive docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Combined status, kept for Render's health check and older callers."""
    return {
        "status": "ok" if MODEL else "degraded",
        "model_loaded": bool(MODEL),
        "model_name": MODEL.get("model_name"),
        "threshold": MODEL.get("threshold"),
        "version": __version__,
    }


@app.get("/health/live", tags=["ops"])
def live() -> dict:
    """The process is up. Says nothing about the model."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["ops"])
def ready() -> JSONResponse:
    """Ready to score: 200 only when a model is loaded, otherwise 503."""
    if not MODEL:
        return JSONResponse(status_code=503, content={"status": "not_ready", "model_loaded": False})
    return JSONResponse(
        content={"status": "ready", "model_loaded": True, "model_version": MODEL.get("model_version")}
    )


@app.get("/model-info", tags=["ops"])
def model_info() -> dict:
    """Which model is serving: version, when it was trained, on what data."""
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    provenance = MODEL.get("provenance") or {}
    return {
        "model_version": MODEL.get("model_version"),
        "model_name": MODEL.get("model_name"),
        "trained_at": provenance.get("trained_at"),
        "threshold": MODEL.get("threshold"),
        "dataset_sha256": (provenance.get("dataset") or {}).get("sha256"),
        "calibration_method": MODEL.get("calibration"),
        "shap_drivers_available": MODEL.get("explainer") is not None,
    }


@app.get("/metrics", tags=["ops"])
def metrics() -> dict:
    """A compact summary of the last training run's held-out results.

    The full offline report stays in ``reports/metrics.json``; this endpoint does
    not serve it.
    """
    if not config.METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="No metrics report found.")
    report = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    test = report.get("test_metrics_tuned_threshold", {})
    impact = report.get("business_impact", {})
    interval = (report.get("uncertainty") or {}).get("profit_model") or {}
    return {
        "model_version": (report.get("provenance") or {}).get("model_version"),
        "selected_model": report.get("selected_model"),
        "calibration_method": report.get("calibration_method"),
        "threshold": test.get("threshold"),
        "test": {
            k: test.get(k) for k in ("roc_auc", "pr_auc", "brier_score", "ece", "precision", "recall")
        },
        "test_campaign_profit": {
            "model": impact.get("net_benefit_optimal"),
            "blanket_campaign": impact.get("net_benefit_blanket_campaign"),
            "do_nothing": 0.0,
            "model_90pct_interval": [interval.get("low"), interval.get("high")] if interval else None,
        },
        "cost_assumptions_not_observed_outcomes": impact.get("assumptions"),
    }


@app.post("/drift", tags=["monitoring"])
def drift_check(request: DriftRequest) -> dict:
    """Compare a recent batch to the reference sample saved at training time.

    Detects and reports only; it never retrains or changes the model. Uses
    SciPy tests, so it works without the optional Evidently package.
    """
    if not MODEL or MODEL.get("drift_reference") is None:
        raise HTTPException(
            status_code=503,
            detail="No drift reference loaded. Run `python -m churnguard.train`, then restart.",
        )
    return drift.analyse(MODEL, pd.DataFrame(request.records))


@app.post("/predict", response_model=Prediction, tags=["scoring"])
def predict(customer: Customer) -> Prediction:
    """Score a single customer."""
    return _score(pd.DataFrame([customer.model_dump()]))[0]


@app.post("/predict/batch", response_model=list[Prediction], tags=["scoring"])
def predict_batch(request: BatchRequest) -> list[Prediction]:
    """Score up to 1,000 customers in one call."""
    return _score(pd.DataFrame([c.model_dump() for c in request.customers]))
