import logging

import pandas as pd

from app import drift_monitor as drift_module
from app.drift_monitor import DriftMonitor
from app.schemas import TOP_SHAP_FEATURES


class FakePredictionStore:
    backend_name = "fake"

    def __init__(self):
        self.drift_events = []

    def record_drift_event(self, **payload):
        self.drift_events.append(payload)
        return True


def build_monitor(reference_data):
    monitor = DriftMonitor.__new__(DriftMonitor)
    monitor.batch_size = 100
    monitor.drift_threshold = 0.05
    monitor.reference_data = reference_data
    monitor.prediction_buffer = []
    monitor.prediction_store = FakePredictionStore()
    monitor.logger = logging.getLogger("drift_monitor")
    return monitor


def test_evidently_summary_marks_drifted_features():
    monitor = build_monitor(pd.DataFrame())

    summary = monitor._summarize_evidently_result(
        {
            "metrics": [
                {
                    "config": {
                        "type": "evidently:metric_v2:DriftedColumnsCount",
                    },
                    "value": {"count": 1.0, "share": 0.5},
                },
                {
                    "metric_name": "ValueDrift(column=AMT_CREDIT,method=ks p_value,threshold=0.05)",
                    "config": {
                        "type": "evidently:metric_v2:ValueDrift",
                        "column": "AMT_CREDIT",
                    },
                    "value": 0.001,
                },
                {
                    "metric_name": "ValueDrift(column=DAYS_BIRTH,method=ks p_value,threshold=0.05)",
                    "config": {
                        "type": "evidently:metric_v2:ValueDrift",
                        "column": "DAYS_BIRTH",
                    },
                    "value": 0.7,
                },
            ]
        },
        monitored_features=["AMT_CREDIT", "DAYS_BIRTH"],
    )

    assert summary["drift_detected"] is True
    assert summary["drifted_features"] == ["AMT_CREDIT"]
    assert summary["drifted_feature_count"] == 1
    assert summary["feature_scores"]["AMT_CREDIT"] == 0.001


def test_evidently_drift_check_records_true_alert(monkeypatch, caplog):
    class FakeReport:
        def __init__(self, *_):
            pass

        def run(self, *, current_data, reference_data):
            assert list(current_data.columns) == ["AMT_CREDIT"]
            assert list(reference_data.columns) == ["AMT_CREDIT"]
            return {
                "metrics": [
                    {
                        "config": {
                            "type": "evidently:metric_v2:DriftedColumnsCount",
                        },
                        "value": {"count": 1.0, "share": 1.0},
                    },
                    {
                        "metric_name": "ValueDrift(column=AMT_CREDIT,method=ks p_value,threshold=0.05)",
                        "config": {
                            "type": "evidently:metric_v2:ValueDrift",
                            "column": "AMT_CREDIT",
                        },
                        "value": 0.001,
                    },
                ]
            }

    monkeypatch.setattr(drift_module, "Report", FakeReport)
    monkeypatch.setattr(drift_module, "DataDriftPreset", lambda: object())
    monitor = build_monitor(pd.DataFrame({"AMT_CREDIT": [100000, 120000, 130000]}))

    caplog.set_level(logging.INFO, logger="drift_monitor")
    monitor._check_drift_evidently(pd.DataFrame({"AMT_CREDIT": [4000000, 3900000, 3800000]}))

    assert any(
        event["drift_detected"] is True and event["method"] == "evidently"
        for event in monitor.prediction_store.drift_events
    )
    assert any("feature_drift_detected" in record.getMessage() for record in caplog.records)


def test_manual_fallback_monitors_all_predict_features():
    reference = pd.DataFrame(
        {
            feature: [0.5] * 30
            for feature in TOP_SHAP_FEATURES
        }
    )
    current = pd.DataFrame(
        {
            feature: [0.5] * 30
            for feature in TOP_SHAP_FEATURES
        }
    )
    current["EXT_SOURCE_MEAN"] = [0.0] * 30

    monitor = build_monitor(reference)
    monitor._check_drift_manual(current)

    drifted_features = {
        event.get("feature")
        for event in monitor.prediction_store.drift_events
        if event.get("drift_detected") is True
    }

    assert "EXT_SOURCE_MEAN" in drifted_features
