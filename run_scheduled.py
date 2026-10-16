"""Entry point for the unattended refresh. See src/scheduled_run.py.

    python run_scheduled.py
    python run_scheduled.py --force --skip-shap
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.scheduled_run import main

if __name__ == "__main__":
    sys.exit(main())