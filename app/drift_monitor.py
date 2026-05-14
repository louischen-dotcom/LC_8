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

    def __init__(self, reference_data_path: str, batch_size: int = 100, drift_threshold: float = 0.05):
        self.reference_data_path = Path(reference_data_path)
        self.batch_size = batch_size
        self.drift_threshold = drift_threshold
        self.reference_data: Optional[pd.DataFrame] = None
        self.prediction_buffer: List[Dict[str, Any]] = []
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
            drift_detected = dataset_drift['dataset_drift']
            drifted_features = dataset_drift['number_of_drifted_columns']
            total_features = dataset_drift['number_of_columns']

            # Log overall result
            self.logger.info(
                json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": "drift_analysis_completed",
                    "method": "evidently",
                    "drift_detected": drift_detected,
                    "drifted_features": drifted_features,
                    "total_features": total_features,
                    "batch_size": len(batch_df)
                })
            )

            # Log per-feature results for drifted features
            if drift_detected:
                drift_table = report_dict['metrics'][1]['result']

                for col, info in drift_table['drift_by_columns'].items():
                    if info['drift_detected']:
                        self.logger.warning(
                            json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "event": "feature_drift_detected",
                                "feature": col,
                                "drift_score": round(info['drift_score'], 6),
                                "stattest_name": info['stattest_name'],
                                "method": "evidently"
                            })
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

                        if p_value < self.drift_threshold:
                            drift_detected = True
                            self.logger.warning(
                                json.dumps({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "event": "feature_drift_detected",
                                    "feature": feature,
                                    "ks_statistic": round(ks_stat, 4),
                                    "p_value": round(p_value, 6),
                                    "method": "manual_ks_test"
                                })
                            )
                except Exception as e:
                    self.logger.error(f"Error checking drift for {feature}: {e}")

        self.logger.info(
            json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "drift_analysis_completed",
                "method": "manual_ks_test",
                "drift_detected": drift_detected,
                "batch_size": len(batch_df)
            })
        )

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get current drift monitoring status."""
        return {
            "reference_data_shape": self.reference_data.shape if self.reference_data is not None else None,
            "buffer_size": len(self.prediction_buffer),
            "batch_size": self.batch_size,
            "drift_threshold": self.drift_threshold,
            "evidently_available": EVIDENTLY_AVAILABLE
        }