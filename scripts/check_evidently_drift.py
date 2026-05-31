import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.drift_monitor import DriftMonitor


def main():
    reference_data_path = PROJECT_ROOT / "data" / "processed" / "train_final.csv"

    monitor = DriftMonitor(
        str(reference_data_path),
        batch_size=30,
    )

    for i in range(30):
        sample = {
            "AMT_INCOME_TOTAL": 100000.0 + i * 1000,
            "AMT_CREDIT": 400000.0 + i * 5000,
            "DAYS_BIRTH": -12000 - i,
            "DAYS_EMPLOYED": -2000 - i,
            "prediction": 0,
            "probability_of_default": 0.42,
            "risk_category": "Medium",
        }
        monitor.add_prediction(sample)

    print("Drift monitor executed")
    print(monitor.get_drift_summary())


if __name__ == "__main__":
    main()