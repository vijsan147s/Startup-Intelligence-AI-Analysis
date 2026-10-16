"""Thin entry point over src.pipeline.run.

All pipeline logic lives in src/pipeline.py so this file and run_full.py cannot
drift apart. See that module for the step order.

    python run_pipeline.py              # everything
    python run_pipeline.py --skip-shap  # faster, SHAP is the slow step
"""

import sys

sys.path.insert(0, '.')

from src.pipeline import run

if __name__ == "__main__":
    run(
        skip_shap="--skip-shap" in sys.argv,
        skip_classification="--skip-classification" in sys.argv,
        skip_clustering="--skip-clustering" in sys.argv,
    )