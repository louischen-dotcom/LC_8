from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from sqlalchemy import create_engine, text


DEFAULT_ERROR_RATE_THRESHOLD = 0.05
DEFAULT_LATENCY_P95_THRESHOLD_MS = 1000.0


def compute_operational_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    error_rate_threshold: float = DEFAULT_ERROR_RATE_THRESHOLD,
    latency_p95_threshold_ms: float = DEFAULT_LATENCY_P95_THRESHOLD_MS,
    min_requests: int = 1,
) -> dict[str, Any]:
    rows = list(records)
    total_requests = len(rows)
    error_count = sum(
        1
        for row in rows
        if row.get("event") == "prediction_error" or bool(row.get("error"))
    )
    success_count = total_requests - error_count
    error_rate = error_count / total_requests if total_requests else 0.0

    latencies = sorted(
        float(row["inference_time_ms"])
        for row in rows
        if row.get("inference_time_ms") is not None
    )

    avg_latency = sum(latencies) / len(latencies) if latencies else None
    p95_latency = percentile_nearest_rank(latencies, 0.95) if latencies else None
    max_latency = latencies[-1] if latencies else None

    alerts = []
    enough_traffic = total_requests >= min_requests

    if enough_traffic and error_rate > error_rate_threshold:
        alerts.append(
            {
                "type": "high_error_rate",
                "message": "Prediction error rate is above threshold",
                "value": round(error_rate, 4),
                "threshold": error_rate_threshold,
            }
        )

    if p95_latency is not None and p95_latency > latency_p95_threshold_ms:
        alerts.append(
            {
                "type": "high_latency_p95",
                "message": "Prediction p95 latency is above threshold",
                "value_ms": round(p95_latency, 2),
                "threshold_ms": latency_p95_threshold_ms,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_requests": total_requests,
        "success_count": success_count,
        "error_count": error_count,
        "error_rate": round(error_rate, 4),
        "latency_ms": {
            "average": round(avg_latency, 2) if avg_latency is not None else None,
            "p95": round(p95_latency, 2) if p95_latency is not None else None,
            "max": round(max_latency, 2) if max_latency is not None else None,
        },
        "thresholds": {
            "error_rate": error_rate_threshold,
            "latency_p95_ms": latency_p95_threshold_ms,
            "min_requests": min_requests,
        },
        "status": "alert" if alerts else "ok",
        "alerts": alerts,
    }


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")

    rank = max(1, int(percentile * len(values) + 0.999999))
    return values[min(rank, len(values)) - 1]


def load_prediction_records(
    database_url: str,
    *,
    lookback_hours: float | None,
    limit: int,
) -> list[dict[str, Any]]:
    engine = create_engine(normalize_database_url(database_url), pool_pre_ping=True)
    since_ts = (
        datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        if lookback_hours
        else None
    )

    if since_ts:
        query = text(
            """
            select timestamp, event, inference_time_ms, error
            from prediction_logs
            where timestamp >= :since_ts
            order by timestamp desc
            limit :limit
            """
        )
        params = {"since_ts": since_ts, "limit": limit}
    else:
        query = text(
            """
            select timestamp, event, inference_time_ms, error
            from prediction_logs
            order by timestamp desc
            limit :limit
            """
        )
        params = {"limit": limit}

    with engine.connect() as connection:
        rows = connection.execute(query, params).mappings().all()

    engine.dispose()
    return [dict(row) for row in rows]


def normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze prediction error rate and latency from prediction_logs.",
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="SQLAlchemy database URL. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--lookback-hours",
        type=float,
        default=float(os.environ.get("OPERATIONAL_LOOKBACK_HOURS", "24")),
        help="Analyze only recent records. Use 0 to disable the lookback filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("OPERATIONAL_ANALYSIS_LIMIT", "10000")),
        help="Maximum number of prediction records to analyze.",
    )
    parser.add_argument(
        "--error-rate-threshold",
        type=float,
        default=float(
            os.environ.get(
                "OPERATIONAL_ERROR_RATE_THRESHOLD",
                DEFAULT_ERROR_RATE_THRESHOLD,
            )
        ),
        help="Alert threshold for prediction error rate.",
    )
    parser.add_argument(
        "--latency-p95-threshold-ms",
        type=float,
        default=float(
            os.environ.get(
                "OPERATIONAL_LATENCY_P95_THRESHOLD_MS",
                DEFAULT_LATENCY_P95_THRESHOLD_MS,
            )
        ),
        help="Alert threshold for p95 prediction latency in milliseconds.",
    )
    parser.add_argument(
        "--min-requests",
        type=int,
        default=int(os.environ.get("OPERATIONAL_MIN_REQUESTS", "1")),
        help="Minimum request count before error-rate alerts are emitted.",
    )
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Exit with code 2 when an alert is detected.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")

    lookback_hours = args.lookback_hours if args.lookback_hours > 0 else None
    records = load_prediction_records(
        args.database_url,
        lookback_hours=lookback_hours,
        limit=args.limit,
    )
    report = compute_operational_metrics(
        records,
        error_rate_threshold=args.error_rate_threshold,
        latency_p95_threshold_ms=args.latency_p95_threshold_ms,
        min_requests=args.min_requests,
    )
    print(json.dumps(report, indent=2))

    if args.fail_on_alert and report["status"] == "alert":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
