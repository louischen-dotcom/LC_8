import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import build_model_array, predict_default_probability
from app.model_loader import load_model
from app.schemas import CreditApplication


def main():
    model = load_model()
    application = CreditApplication()
    model_input = build_model_array(application)

    warmup_runs = 50
    measured_runs = 1000

    for _ in range(warmup_runs):
        predict_default_probability(model, model_input)

    times_ms = []

    for _ in range(measured_runs):
        start = time.perf_counter()
        predict_default_probability(model, model_input)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed_ms)

    print("Optimized inference benchmark")
    print(f"Runs: {measured_runs}")
    print(f"Mean latency: {statistics.mean(times_ms):.4f} ms")
    print(f"Median latency: {statistics.median(times_ms):.4f} ms")
    print(f"Min latency: {min(times_ms):.4f} ms")
    print(f"Max latency: {max(times_ms):.4f} ms")


if __name__ == "__main__":
    main()
