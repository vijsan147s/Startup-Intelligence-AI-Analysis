"""Alias for run_pipeline.py.

Kept for backwards compatibility with the earlier README instructions. Both
files now delegate to src/pipeline.py, so there is one implementation.

    python run_full.py                  # everything
    python run_full.py --skip-shap      # faster
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