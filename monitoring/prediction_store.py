from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Protocol

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from monitoring.db_models import Base, DriftEvent, PredictionLog


logger = logging.getLogger(__name__)


class PredictionStore(Protocol):
    backend_name: str

    def record_prediction(
        self,
        *,
        timestamp: datetime,
        input_data: dict[str, Any],
        prediction: int,
        probability_of_default: float,
        risk_category: str,
        inference_time_ms: float,
        model_version: str | None = None,
    ) -> bool:
        ...

    def record_prediction_error(
        self,
        *,
        timestamp: datetime,
        input_data: dict[str, Any],
        error: str,
        inference_time_ms: float | None = None,
        model_version: str | None = None,
    ) -> bool:
        ...

    def record_drift_event(
        self,
        *,
        timestamp: datetime,
        drift_detected: bool | None = None,
        feature: str | None = None,
        drift_score: float | None = None,
        p_value: float | None = None,
        ks_statistic: float | None = None,
        method: str | None = None,
        batch_size: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> bool:
        ...

    def close(self) -> None:
        ...


class DisabledPredictionStore:
    backend_name = "disabled"

    def record_prediction(self, **_: Any) -> bool:
        return False

    def record_prediction_error(self, **_: Any) -> bool:
        return False

    def record_drift_event(self, **_: Any) -> bool:
        return False

    def close(self) -> None:
        return None


class SqlAlchemyPredictionStore:
    backend_name = "sqlalchemy"

    def __init__(self, database_url: str, auto_create_tables: bool = True):
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

        if auto_create_tables:
            Base.metadata.create_all(self.engine)

    def record_prediction(
        self,
        *,
        timestamp: datetime,
        input_data: dict[str, Any],
        prediction: int,
        probability_of_default: float,
        risk_category: str,
        inference_time_ms: float,
        model_version: str | None = None,
    ) -> bool:
        return self._insert(
            PredictionLog(
                timestamp=timestamp,
                event="prediction",
                input_data=input_data,
                prediction=prediction,
                probability_of_default=probability_of_default,
                risk_category=risk_category,
                inference_time_ms=inference_time_ms,
                model_version=model_version,
            )
        )

    def record_prediction_error(
        self,
        *,
        timestamp: datetime,
        input_data: dict[str, Any],
        error: str,
        inference_time_ms: float | None = None,
        model_version: str | None = None,
    ) -> bool:
        return self._insert(
            PredictionLog(
                timestamp=timestamp,
                event="prediction_error",
                input_data=input_data,
                inference_time_ms=inference_time_ms,
                error=error,
                model_version=model_version,
            )
        )

    def record_drift_event(
        self,
        *,
        timestamp: datetime,
        drift_detected: bool | None = None,
        feature: str | None = None,
        drift_score: float | None = None,
        p_value: float | None = None,
        ks_statistic: float | None = None,
        method: str | None = None,
        batch_size: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> bool:
        return self._insert(
            DriftEvent(
                timestamp=timestamp,
                drift_detected=drift_detected,
                feature=feature,
                drift_score=drift_score,
                p_value=p_value,
                ks_statistic=ks_statistic,
                method=method,
                batch_size=batch_size,
                details=details,
            )
        )

    def _insert(self, record: PredictionLog | DriftEvent) -> bool:
        try:
            with self.session_factory() as session:
                session.add(record)
                session.commit()
            return True
        except SQLAlchemyError as exc:
            logger.warning("Failed to persist monitoring record: %s", exc)
            return False

    def close(self) -> None:
        self.engine.dispose()


class SupabaseRestPredictionStore:
    backend_name = "supabase_rest"

    def __init__(
        self,
        *,
        supabase_url: str,
        api_key: str,
        prediction_table: str = "prediction_logs",
        drift_table: str = "drift_events",
        timeout_seconds: float = 5.0,
    ):
        self.supabase_url = supabase_url.rstrip("/")
        self.api_key = api_key
        self.prediction_table = prediction_table
        self.drift_table = drift_table
        self.timeout_seconds = timeout_seconds

    def record_prediction(
        self,
        *,
        timestamp: datetime,
        input_data: dict[str, Any],
        prediction: int,
        probability_of_default: float,
        risk_category: str,
        inference_time_ms: float,
        model_version: str | None = None,
    ) -> bool:
        return self._post(
            self.prediction_table,
            {
                "timestamp": timestamp.isoformat(),
                "event": "prediction",
                "input_data": input_data,
                "prediction": prediction,
                "probability_of_default": probability_of_default,
                "risk_category": risk_category,
                "inference_time_ms": inference_time_ms,
                "model_version": model_version,
            },
        )

    def record_prediction_error(
        self,
        *,
        timestamp: datetime,
        input_data: dict[str, Any],
        error: str,
        inference_time_ms: float | None = None,
        model_version: str | None = None,
    ) -> bool:
        return self._post(
            self.prediction_table,
            {
                "timestamp": timestamp.isoformat(),
                "event": "prediction_error",
                "input_data": input_data,
                "inference_time_ms": inference_time_ms,
                "error": error,
                "model_version": model_version,
            },
        )

    def record_drift_event(
        self,
        *,
        timestamp: datetime,
        drift_detected: bool | None = None,
        feature: str | None = None,
        drift_score: float | None = None,
        p_value: float | None = None,
        ks_statistic: float | None = None,
        method: str | None = None,
        batch_size: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> bool:
        return self._post(
            self.drift_table,
            {
                "timestamp": timestamp.isoformat(),
                "drift_detected": drift_detected,
                "feature": feature,
                "drift_score": drift_score,
                "p_value": p_value,
                "ks_statistic": ks_statistic,
                "method": method,
                "batch_size": batch_size,
                "details": details,
            },
        )

    def _post(self, table: str, payload: dict[str, Any]) -> bool:
        url = f"{self.supabase_url}/rest/v1/{table}"
        body = json.dumps(_drop_none(payload)).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "apikey": self.api_key,
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, TimeoutError) as exc:
            logger.warning("Failed to persist monitoring record to Supabase: %s", exc)
            return False

    def close(self) -> None:
        return None


def get_prediction_store() -> PredictionStore:
    backend = os.environ.get("MONITORING_BACKEND", "auto").lower()
    database_url = os.environ.get("DATABASE_URL")
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
    )

    if backend in {"off", "disabled", "none"}:
        return DisabledPredictionStore()

    if backend in {"supabase", "supabase_rest"} or (
        backend == "auto" and not database_url and supabase_url and supabase_key
    ):
        if not supabase_url or not supabase_key:
            logger.warning("Supabase monitoring backend selected but credentials are missing")
            return DisabledPredictionStore()
        return SupabaseRestPredictionStore(
            supabase_url=supabase_url,
            api_key=supabase_key,
            prediction_table=os.environ.get("SUPABASE_PREDICTION_TABLE", "prediction_logs"),
            drift_table=os.environ.get("SUPABASE_DRIFT_TABLE", "drift_events"),
        )

    if database_url:
        try:
            auto_create_tables = os.environ.get(
                "MONITORING_AUTO_CREATE_TABLES",
                "true",
            ).lower() not in {"0", "false", "no"}
            return SqlAlchemyPredictionStore(
                _normalize_database_url(database_url),
                auto_create_tables=auto_create_tables,
            )
        except SQLAlchemyError as exc:
            logger.warning("SQLAlchemy monitoring backend unavailable: %s", exc)

    return DisabledPredictionStore()


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url
