from monitoring.analyze_operational_metrics import compute_operational_metrics


def test_operational_metrics_reports_ok_when_under_thresholds():
    report = compute_operational_metrics(
        [
            {"event": "prediction", "inference_time_ms": 100},
            {"event": "prediction", "inference_time_ms": 150},
            {"event": "prediction", "inference_time_ms": 200},
        ],
        error_rate_threshold=0.05,
        latency_p95_threshold_ms=500,
    )

    assert report["status"] == "ok"
    assert report["total_requests"] == 3
    assert report["error_rate"] == 0
    assert report["latency_ms"]["p95"] == 200
    assert report["alerts"] == []


def test_operational_metrics_alerts_on_high_error_rate():
    report = compute_operational_metrics(
        [
            {"event": "prediction", "inference_time_ms": 100},
            {"event": "prediction_error", "inference_time_ms": 120, "error": "boom"},
        ],
        error_rate_threshold=0.05,
        latency_p95_threshold_ms=500,
    )

    assert report["status"] == "alert"
    assert report["error_count"] == 1
    assert report["error_rate"] == 0.5
    assert report["alerts"][0]["type"] == "high_error_rate"


def test_operational_metrics_alerts_on_high_latency_p95():
    report = compute_operational_metrics(
        [
            {"event": "prediction", "inference_time_ms": 100},
            {"event": "prediction", "inference_time_ms": 200},
            {"event": "prediction", "inference_time_ms": 1200},
        ],
        error_rate_threshold=0.05,
        latency_p95_threshold_ms=1000,
    )

    assert report["status"] == "alert"
    assert report["latency_ms"]["p95"] == 1200
    assert report["alerts"][0]["type"] == "high_latency_p95"
