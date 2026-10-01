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


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """Readable 422s: field + message only.

    The default body echoes the rejected input, which crashes the response for
    NaN/Infinity (not valid JSON) and would also echo customer data back.
    """
    detail = [
        {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


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

    probabilities = MODEL["pipeline"].predict_proba(frame)[:, 1]
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
    """Liveness plus which artifact is actually in memory."""
    return {
        "status": "ok" if MODEL else "degraded",
        "model_loaded": bool(MODEL),
        "model_name": MODEL.get("model_name"),
        "threshold": MODEL.get("threshold"),
        "version": __version__,
    }


@app.get("/metrics", tags=["ops"])
def metrics() -> dict:
    """The full evaluation report produced by the last training run."""
    if not config.METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="No metrics report found.")
    return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))


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
