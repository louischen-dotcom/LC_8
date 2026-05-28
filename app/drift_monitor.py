# /app/drift_monitor.py

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
from datetime import datetime, timezone
import json

logger = logging.getLogger(__name__)

try:
    from evidently.report import Report
    from evidently.metrics import DatasetDriftMetric, DataDriftTable
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

        # Check if we should run drift analysis
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
        """Use Evidently for comprehensive drift analysis."""
        try:
            # Create Evidently report
            drift_report = Report(metrics=[
                DatasetDriftMetric(),
                DataDriftTable(),
            ])

            # Run the report
            drift_report.run(
                reference_data=self.reference_data,
                current_data=batch_df,
            )

            # Extract results
            report_dict = drift_report.as_dict()

            # Overall drift result
            dataset_drift = report_dict['metrics'][0]['result']
            drift_detected = bool(dataset_drift['dataset_drift'])
            drifted_features = int(dataset_drift['number_of_drifted_columns'])
            total_features = int(dataset_drift['number_of_columns'])
            event_timestamp = datetime.now(timezone.utc)

            # Log overall result
            self.logger.info(
                json.dumps({
                    "timestamp": event_timestamp.isoformat(),
                    "event": "drift_analysis_completed",
                    "method": "evidently",
                    "drift_detected": drift_detected,
                    "drifted_features": drifted_features,
                    "total_features": total_features,
                    "batch_size": len(batch_df)
                })
            )
            self._record_drift_event(
                timestamp=event_timestamp,
                drift_detected=drift_detected,
                method="evidently",
                batch_size=len(batch_df),
                details={
                    "drifted_features": drifted_features,
                    "total_features": total_features,
                },
            )

            # Log per-feature results for drifted features
            if drift_detected:
                drift_table = report_dict['metrics'][1]['result']

                for col, info in drift_table['drift_by_columns'].items():
                    if info['drift_detected']:
                        event_timestamp = datetime.now(timezone.utc)
                        drift_score = float(info['drift_score'])
                        self.logger.warning(
                            json.dumps({
                                "timestamp": event_timestamp.isoformat(),
                                "event": "feature_drift_detected",
                                "feature": col,
                                "drift_score": round(drift_score, 6),
                                "stattest_name": info['stattest_name'],
                                "method": "evidently"
                            })
                        )
                        self._record_drift_event(
                            timestamp=event_timestamp,
                            drift_detected=True,
                            feature=col,
                            drift_score=drift_score,
                            method="evidently",
                            batch_size=len(batch_df),
                            details={"stattest_name": info.get("stattest_name")},
                        )

        except Exception as e:
            self.logger.error(f"Evidently drift analysis failed: {e}")
            # Fallback to manual method
            self._check_drift_manual(batch_df)

    def _check_drift_manual(self, batch_df: pd.DataFrame):
        """Fallback manual drift detection using KS test."""
        # Key features to monitor for drift
        key_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_BIRTH', 'DAYS_EMPLOYED']

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
