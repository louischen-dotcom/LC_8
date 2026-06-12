# /app/drift_monitor.py

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
from datetime import datetime, timezone
import json

from app.schemas import TOP_SHAP_FEATURES

logger = logging.getLogger(__name__)

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently not available, falling back to manual drift detection")


class DriftMonitor:
    """Monitor data drift in production predictions against training data."""

    def __init__(
        self,
        reference_data_path: str,
        batch_size: int = 100,
        drift_threshold: float = 0.05,
        prediction_store: Any | None = None,
    ):
        self.reference_data_path = Path(reference_data_path)
        self.batch_size = batch_size
        self.drift_threshold = drift_threshold
        self.reference_data: Optional[pd.DataFrame] = None
        self.prediction_buffer: List[Dict[str, Any]] = []
        self.prediction_store = prediction_store
        self.logger = logging.getLogger("drift_monitor")

        # Load reference data on initialization
        self._load_reference_data()

    def _load_reference_data(self):
        """Load reference (training) data for drift comparison."""
        try:
            self.reference_data = pd.read_csv(self.reference_data_path)
            self.logger.info(f"Loaded reference data: {self.reference_data.shape}")
        except Exception as e:
            self.logger.error(f"Failed to load reference data: {e}")
            raise

    def add_prediction(self, prediction_data: Dict[str, Any]):
        """Add a prediction to the monitoring buffer."""
        self.prediction_buffer.append(prediction_data)

        if len(self.prediction_buffer) >= self.batch_size:
            self._check_drift()

    def _check_drift(self):
        """Run drift analysis on accumulated predictions using Evidently or fallback."""
        if self.reference_data is None:
            return

        # Convert buffer to DataFrame
        batch_df = pd.DataFrame(self.prediction_buffer)

        if EVIDENTLY_AVAILABLE:
            self._check_drift_evidently(batch_df)
        else:
            self._check_drift_manual(batch_df)

        # Clear buffer after analysis
        self.prediction_buffer.clear()

    def _check_drift_evidently(self, batch_df: pd.DataFrame):
        """Use Evidently for drift analysis."""
        try:
            common_columns = [
                col for col in batch_df.columns
                if col in self.reference_data.columns
            ]

            if not common_columns:
                self.logger.warning("No common columns available for Evidently drift analysis")
                self._check_drift_manual(batch_df)
                return

            current_data = batch_df[common_columns]
            reference_data = self.reference_data[common_columns]

            drift_report = Report([
                DataDriftPreset()
            ])

            drift_result = drift_report.run(
                current_data=current_data,
                reference_data=reference_data,
            )
            drift_summary = self._summarize_evidently_result(
                drift_result,
                monitored_features=common_columns,
            )

            event_timestamp = datetime.now(timezone.utc)

            self.logger.info(
                json.dumps({
                    "timestamp": event_timestamp.isoformat(),
                    "event": "drift_analysis_completed",
                    "method": "evidently",
                    "drift_detected": drift_summary["drift_detected"],
                    "drifted_features": drift_summary["drifted_features"],
                    "drifted_feature_count": drift_summary["drifted_feature_count"],
                    "drifted_feature_share": drift_summary["drifted_feature_share"],
                    "batch_size": len(batch_df),
                    "monitored_features": common_columns,
                })
            )

            for feature in drift_summary["drifted_features"]:
                self.logger.warning(
                    json.dumps({
                        "timestamp": event_timestamp.isoformat(),
                        "event": "feature_drift_detected",
                        "feature": feature,
                        "method": "evidently",
                    })
                )
                self._record_drift_event(
                    timestamp=event_timestamp,
                    drift_detected=True,
                    feature=feature,
                    method="evidently",
                    batch_size=len(batch_df),
                    details={
                        "drift_score": drift_summary["feature_scores"].get(feature),
                    },
                )

            self._record_drift_event(
                timestamp=event_timestamp,
                drift_detected=drift_summary["drift_detected"],
                method="evidently",
                batch_size=len(batch_df),
                details={
                    "monitored_features": common_columns,
                    **drift_summary,
                },
            )

        except Exception as e:
            self.logger.error(f"Evidently drift analysis failed: {e}")
            self._check_drift_manual(batch_df)

    def _check_drift_manual(self, batch_df: pd.DataFrame):
        """Fallback manual drift detection using KS test."""
        key_features = list(TOP_SHAP_FEATURES)

        drift_detected = False

        for feature in key_features:
            if feature in batch_df.columns and feature in self.reference_data.columns:
                try:
                    ref_data = self.reference_data[feature].dropna()
                    prod_data = batch_df[feature].dropna()

                    if len(prod_data) > 10:  # Need minimum sample size
                        ks_stat, p_value = stats.ks_2samp(ref_data, prod_data)
                        ks_stat_value = float(ks_stat)
                        p_value_value = float(p_value)

                        if p_value_value < self.drift_threshold:
                            drift_detected = True
                            event_timestamp = datetime.now(timezone.utc)
                            self.logger.warning(
                                json.dumps({
                                    "timestamp": event_timestamp.isoformat(),
                                    "event": "feature_drift_detected",
                                    "feature": feature,
                                    "ks_statistic": round(ks_stat_value, 4),
                                    "p_value": round(p_value_value, 6),
                                    "method": "manual_ks_test"
                                })
                            )
                            self._record_drift_event(
                                timestamp=event_timestamp,
                                drift_detected=True,
                                feature=feature,
                                p_value=p_value_value,
                                ks_statistic=ks_stat_value,
                                method="manual_ks_test",
                                batch_size=len(batch_df),
                            )
                except Exception as e:
                    self.logger.error(f"Error checking drift for {feature}: {e}")

        event_timestamp = datetime.now(timezone.utc)
        self.logger.info(
            json.dumps({
                "timestamp": event_timestamp.isoformat(),
                "event": "drift_analysis_completed",
                "method": "manual_ks_test",
                "drift_detected": drift_detected,
                "batch_size": len(batch_df)
            })
        )
        self._record_drift_event(
            timestamp=event_timestamp,
            drift_detected=drift_detected,
            method="manual_ks_test",
            batch_size=len(batch_df),
        )

    def _summarize_evidently_result(
        self,
        drift_result: Any,
        *,
        monitored_features: list[str],
    ) -> dict[str, Any]:
        """Extract a stable drift summary from Evidently's report snapshot."""
        result_dict = self._evidently_result_to_dict(drift_result)
        metrics = result_dict.get("metrics", [])

        drifted_feature_count = 0
        drifted_feature_share = 0.0
        drifted_features: set[str] = set()
        feature_scores: dict[str, float] = {}

        for metric in metrics:
            metric_name = str(metric.get("metric_name", ""))
            config = metric.get("config") or {}
            value = metric.get("value")

            if config.get("type") == "evidently:metric_v2:DriftedColumnsCount":
                if isinstance(value, dict):
                    drifted_feature_count = int(float(value.get("count") or 0))
                    drifted_feature_share = float(value.get("share") or 0.0)
                continue

            feature = config.get("column") or self._feature_from_metric_name(metric_name)
            if feature not in monitored_features:
                continue

            score = self._as_float(value)
            if score is None:
                continue

            feature_scores[feature] = score
            if score < self.drift_threshold:
                drifted_features.add(feature)

        if drifted_features:
            drift_detected = True
        else:
            drift_detected = drifted_feature_count > 0

        return {
            "drift_detected": drift_detected,
            "drifted_features": sorted(drifted_features),
            "drifted_feature_count": max(drifted_feature_count, len(drifted_features)),
            "drifted_feature_share": drifted_feature_share,
            "feature_scores": feature_scores,
        }

    @staticmethod
    def _evidently_result_to_dict(drift_result: Any) -> dict[str, Any]:
        if isinstance(drift_result, dict):
            return drift_result
        for method_name in ("dict", "dump_dict"):
            method = getattr(drift_result, method_name, None)
            if callable(method):
                return method()
        json_method = getattr(drift_result, "json", None)
        if callable(json_method):
            return json.loads(json_method())
        return {}

    @staticmethod
    def _feature_from_metric_name(metric_name: str) -> str | None:
        prefix = "ValueDrift(column="
        if not metric_name.startswith(prefix):
            return None
        return metric_name.removeprefix(prefix).split(",", maxsplit=1)[0]

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if isinstance(value, dict):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _record_drift_event(self, **payload: Any):
        prediction_store = getattr(self, "prediction_store", None)
        if prediction_store:
            try:
                prediction_store.record_drift_event(**payload)
            except Exception as exc:
                self.logger.warning(f"Failed to persist drift event: {exc}")

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get current drift monitoring status."""
        return {
            "reference_data_shape": self.reference_data.shape if self.reference_data is not None else None,
            "buffer_size": len(self.prediction_buffer),
            "batch_size": self.batch_size,
            "drift_threshold": self.drift_threshold,
            "evidently_available": EVIDENTLY_AVAILABLE,
            "monitoring_store_backend": getattr(
                getattr(self, "prediction_store", None),
                "backend_name",
                None,
            ),
        }
