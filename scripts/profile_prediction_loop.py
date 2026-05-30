import cProfile
import pstats
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import build_model_frame, predict_default_probability
from app.model_loader import load_model
from app.schemas import CreditApplication


def main():
    model = load_model()
    application = CreditApplication()
    model_frame = build_model_frame(application)

    for _ in range(10):
        predict_default_probability(model, model_frame)

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(1000):
        predict_default_probability(model, model_frame)

    profiler.disable()

    stats_path = PROJECT_ROOT / "reports" / "profile_prediction_loop.prof"
    stats_path.parent.mkdir(exist_ok=True)

    profiler.dump_stats(stats_path)

    stats = pstats.Stats(str(stats_path))
    stats.strip_dirs().sort_stats("cumtime").print_stats(20)


if __name__ == "__main__":
    main()