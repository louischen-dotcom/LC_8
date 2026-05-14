import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from dotenv import load_dotenv

from app.model_loader import get_model, load_model
from app.schemas import (
    CreditApplication,
    HealthResponse,
    MODEL_FEATURES,
    PredictionResponse,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lc-8_credit_scoring_api")
bearer_scheme = HTTPBearer(auto_error=False)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def build_model_frame(application: CreditApplication) -> pd.DataFrame:
    model_input = application.to_model_input()
    return pd.DataFrame([model_input], columns=MODEL_FEATURES)


def risk_category(probability_of_default: float) -> Literal["Low", "Medium", "High"]:
    if probability_of_default < 0.3:
        return "Low"
    if probability_of_default < 0.6:
        return "Medium"
    return "High"


def predict_default_probability(model, model_frame: pd.DataFrame) -> tuple[int, float]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_frame)
        probability = float(probabilities[0][1])
        prediction = int(probability >= 0.5)
        return prediction, probability

    predictions = model.predict(model_frame)
    prediction = int(predictions[0])
    return prediction, float(prediction)


def verify_api_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    expected_token = os.environ.get("API_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API_TOKEN is not configured",
        )

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(credentials.credentials, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up - loading model")
    load_model()
    logger.info("Model loaded")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Home Credit Scoring API",
    description=(
        "Predict loan default risk from the Top 20 SHAP features. "
        "Hidden model features are imputed with training medians."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": "Home Credit Scoring API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    try:
        get_model()
    except RuntimeError:
        return HealthResponse(status="unhealthy", model_loaded=False)
    return HealthResponse(status="healthy", model_loaded=True)


@app.get("/features", response_model=list[str])
async def exposed_features() -> list[str]:
    return list(CreditApplication.exposed_feature_names())


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_token)],
)
async def predict(application: CreditApplication) -> PredictionResponse:
    start_time = time.perf_counter()

    try:
        model = get_model()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    try:
        model_frame = build_model_frame(application)
        prediction, probability = predict_default_probability(model, model_frame)
        category = risk_category(probability)
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        response = PredictionResponse(
            prediction=prediction,
            probability_of_default=round(probability, 4),
            risk_category=category,
        )

        logger.info(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": "prediction",
                    "exposed_inputs": application.model_dump(),
                    "outputs": response.model_dump(),
                    "inference_time_ms": round(inference_time_ms, 2),
                }
            )
        )

        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": "prediction_error",
                    "error": str(exc),
                    "exposed_inputs": application.model_dump(),
                }
            )
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc
