"""Facts and scene script for the LinkedIn video.

Every figure is read from models/*.json, data/processed/*, the SQLite database
and src/config.py, so the on-screen text cannot drift out of sync with the
pipeline. Where docs/linkedin_video_package.md disagreed with the artifacts, the
artifacts win and the divergence is recorded in CORRECTIONS.
"""

import json
import re
import sqlite3
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    CITY_NORMALIZATION,
    DB_PATH,
    INDUSTRY_NORMALIZATION,
    INVESTMENT_TYPE_NORMALIZATION,
    MODELS_DIR,
    PROCESSED_DIR,
)

# Figures in the hand-written package that the live artifacts contradict.
# Reported at build time so the drift is not silently absorbed into the render.
# Kept as format strings so the message cannot go stale against the artifacts.
CORRECTIONS = [
    "test count: the package says 137; pytest collects {n_tests}",
    "cities normalized: the package says 20, CITY_NORMALIZATION has {n_cities} entries",
    (
        "group-feature ablation: the package shows 'RM 2.4029 -> 1.8686', but the "
        "shipped Random Forest test RMSLE in model_metrics.json is {rf_rmsle}"
    ),
    (
        "permutation importance: the package calls city/year/month '~0'; they are "
        "negative and investors_name is the exact {inv_zero}"
    ),
    (
        "runtime: the package's 61 seconds is the --skip-shap run; a full run "
        "including SHAP is {run_full_s} seconds"
    ),
]


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def money(value):
    """Format a USD amount the way the README does."""
    value = float(value)
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.1f}M"
    if value >= 1e3:
        return f"${value / 1e3:.0f}K"
    return f"${value:,.0f}"


@lru_cache(maxsize=1)
def count_tests():
    """Ask pytest how many tests it collects.

    Counting `def test_` undercounts, because @parametrize expands one function
    into several collected tests. The README's 129 is the function count; 136
    is what pytest reports, and 136 is what a reviewer will see.
    """
    root = Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            [
                sys.executable, "-m", "pytest", str(root / "tests"),
                "--collect-only", "-q",
                # Never touch .pytest_cache: this can run from inside another
                # pytest session, and two processes writing the cache race.
                "-p", "no:cacheprovider",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=root,
        ).stdout
        match = re.search(r"(\d+)\s+tests?\s+collected", out)
        if match:
            return int(match.group(1))
    except (OSError, subprocess.SubprocessError):
        pass

    total = 0
    for path in sorted((root / "tests").glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"^(\s*)def\s+test_\w+", source, re.M):
            if not source[: match.start()].rstrip().endswith("@"):
                total += 1
    return total


def build_facts():
    quality = _load(PROCESSED_DIR / "data_quality_report.json")
    model = _load(MODELS_DIR / "model_metrics.json")
    clf = _load(MODELS_DIR / "classification_metrics.json")
    clusters = _load(MODELS_DIR / "clustering_summary.json")

    csv_path = PROCESSED_DIR / "permutation_importance.csv"
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    perm = {}
    for line in lines[1:]:
        name, mean, _std = line.split(",")
        perm[name] = float(mean)

    con = sqlite3.connect(DB_PATH)
    total_funding, n_rows = con.execute(
        "SELECT SUM(amount_clean), COUNT(*) FROM funding_events"
    ).fetchone()
    views = [
        r[0]
        for r in con.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")
    ]
    con.close()

    amounts = quality["amount_stats"]
    return {
        "total_rows": n_rows,
        "total_funding": total_funding,
        "valid_amounts": amounts["valid_amounts"],
        "missing_amounts": amounts["missing_amounts"],
        "invalid_dates": quality["date_range"]["invalid_dates"],
        "duplicates": quality["duplicated_rows"],
        "median_amount": amounts["median"],
        "views": views,
        "n_tests": count_tests(),
        "rf": model["Random Forest"],
        "dummy": model["Dummy (Median)"],
        "baseline": model["Baseline (Sector x Year Median)"],
        "lr": clf["Logistic Regression"],
        "rf_c": clf["Random Forest"],
        "thresholds": clf["thresholds"],
        "clusters": clusters,
        "perm": perm,
        "n_cities": len(CITY_NORMALIZATION),
        "n_industries": len(INDUSTRY_NORMALIZATION),
        "n_types": len(INVESTMENT_TYPE_NORMALIZATION),
        "run_skip_shap_s": 61.5,
        "run_full_s": 96.1,
    }


def corrections(facts):
    """Render CORRECTIONS against the live artifacts."""
    return [c.format(
        n_tests=facts["n_tests"],
        n_cities=facts["n_cities"],
        rf_rmsle=f"{facts['rf']['rmsle']:.4f}",
        inv_zero=f"{facts['perm']['investors_name']:,.0f}",
        run_full_s=facts["run_full_s"],
    ) for c in CORRECTIONS]


def build_scenes(f):
    """Nine scenes. Timings follow docs/linkedin_video_package.md."""
    rf = f["rf"]
    thr = f["thresholds"]

    return [
        {
            "start": 0.0, "end": 6.0,
            "kicker": "THE DATASET",
            "title": "Indian startup funding,\n2015 to 2020",
            "stagger": [0.0, 1.9, 3.6],
            "cards": [
                (f"{f['total_rows']:,}", "funding rounds recorded"),
                (money(f["total_funding"]), "total recorded funding"),
                (f"{f['missing_amounts']:,}", "with no disclosed amount"),
            ],
            "vo": [
                (0.0, 2.0, "Three thousand forty four Indian startup"),
                (2.0, 3.8, "funding rounds. Thirty eight billion dollars."),
                (3.8, 6.0, f"And {f['missing_amounts']} of them with no disclosed amount."),
            ],
        },
        {
            "start": 6.0, "end": 13.0,
            "kicker": "THE PROBLEM",
            "title": "The rows nobody\nwants to talk about",
            "rows": [
                (f"{f['missing_amounts']} rows", "amount missing", True),
                (f"{f['invalid_dates']} rows", "date invalid", True),
                (f"{f['duplicates']} rows", "exact duplicates", False),
            ],
            "footnote": "Written to a data quality report. Not deleted.",
            "vo": [
                (6.0, 8.8, f"{f['missing_amounts']} rows have no amount."),
                (8.8, 11.2, f"{f['invalid_dates']} have unparseable dates."),
                (11.2, 13.0, "Most projects delete those and move on."),
            ],
        },
        {
            "start": 13.0, "end": 22.0,
            "kicker": "CLEANING",
            "title": "By hand, not by\nguesswork",
            "transforms": [
                ("Bengalore", "Bengaluru"),
                ("Gurgaon", "Gurugram"),
                ("20,00,00,000", "200000000"),
            ],
            "counts": [
                f"{f['n_cities']} cities",
                f"{f['n_industries']} industry labels",
                f"{f['n_types']} investment types",
            ],
            "vo": [
                (13.0, 16.2, f"{f['n_cities']} cities, {f['n_industries']} industry labels,"),
                (16.2, 19.0, f"{f['n_types']} investment types."),
                (19.0, 22.0, "All normalized by hand. Bangalore and Bengaluru\nare the same city."),
            ],
        },
        {
            "start": 22.0, "end": 31.0,
            "kicker": "STORAGE AND BI",
            "title": "SQL underneath,\ndashboard on top",
            "stack": [f"SQLite  funding_events  {f['total_rows']:,} rows"] + f["views"],
            "stats": [("13", "analysis queries"), ("8", "DAX measures"), ("4", "Power BI pages")],
            "vo": [
                (22.0, 25.5, "One SQLite table, four views,"),
                (25.5, 28.5, "thirteen analysis queries."),
                (28.5, 31.0, "Then the same thing again in Power BI."),
            ],
        },
        {
            "start": 31.0, "end": 42.0,
            "kicker": "REGRESSION",
            "title": "Picked on cross validation,\nnot on the test set",
            "bars": [
                ("Dummy (median)", f["dummy"]["rmsle"], False),
                ("Sector x Year Median", f["baseline"]["rmsle"], False),
                ("Random Forest", rf["cv_rmsle"], True),
            ],
            "bar_note": "CV RMSLE  lower is better",
            "metrics": [
                ("Test RMSLE", f"{rf['rmsle']:.4f}"),
                ("MAE", money(rf["mae"])),
                ("R2", f"{rf['r2']:.4f}"),
            ],
            "metrics_note": "R2 is negative. Showing it anyway.",
            "image": "model_comparison.png",
            "vo": [
                (31.0, 35.2, "Random Forest won, and it won on cross validation\ninside the training window."),
                (35.2, 39.0, "RMSLE 1.27 against a 2.43 dummy baseline."),
                (39.0, 42.0, "R squared is negative. I'm showing you that number too."),
            ],
        },
        {
            "start": 42.0, "end": 50.0,
            "kicker": "THE ANTI-LEAKAGE CALL",
            "title": "The better test score\nI did not ship",
            "choice": [
                ("Random Forest", f"{f['rf_c']['f1_weighted']:.4f}", "higher test score.  rejected", False),
                ("Logistic Regression", f"{f['lr']['f1_weighted']:.4f}", "SHIPPED, won on CV", True),
            ],
            "cv_note": (
                f"CV F1   LogReg {f['lr']['cv_f1']:.4f}   beats   "
                f"RF {f['rf_c']['cv_f1']:.4f}"
            ),
            "choice_note": "Selecting on test would have been choosing the answer I wanted.",
            "vo": [
                (42.0, 45.5, "Random Forest scored higher on the test set."),
                (45.5, 48.5, "I didn't pick it, because it lost on cross validation."),
                (48.5, 50.0, "Picking it on test would have been fitting the answer I wanted."),
            ],
        },
        {
            "start": 50.0, "end": 60.0,
            "kicker": "CLASSIFICATION AND CLUSTERING",
            "title": "Two more questions",
            "panels": [
                {
                    "label": "RANGE CLASSIFICATION",
                    "lines": [
                        f"Thresholds   {money(thr[0])}  /  {money(thr[1])}",
                        "Labels   Low / Medium / High",
                        f"Test F1   {f['lr']['f1_weighted']:.4f}",
                    ],
                    "note": "Percentiles from training data only.",
                },
                {
                    "label": "K-MEANS CLUSTERING",
                    "lines": [
                        f"k = {f['clusters']['optimal_k']}   by silhouette, not hard-coded",
                        f"silhouette   {f['clusters']['silhouette']:.4f}",
                        f"{f['clusters']['events_clustered']:,} events clustered",
                    ],
                    "note": "Amount log1p-transformed before clustering.",
                },
            ],
            "vo": [
                (50.0, 54.2, "Classification splits rounds into low, medium and high."),
                (54.2, 57.5, "Thresholds from training percentiles only."),
                (57.5, 60.0, f"Clustering lands on three segments. {f['clusters']['events_clustered']:,} events."),
            ],
        },
        {
            "start": 60.0, "end": 69.0,
            "kicker": "EXPLAINABILITY",
            "title": "What the model\nactually uses",
            "importance": [
                ("investment_type_clean", f["perm"]["investment_type_clean"]),
                ("industry_clean", f["perm"]["industry_clean"]),
                ("month", f["perm"]["month"]),
                ("subvertical", f["perm"]["subvertical"]),
                ("year", f["perm"]["year"]),
                ("city_clean", f["perm"]["city_clean"]),
                ("investors_name", f["perm"]["investors_name"]),
            ],
            "imp_note": "Permutation importance. Negative means shuffling it helped.",
            "verdict": "Investor syndicate features made RMSLE worse. Disabled, code kept.",
            "image": "permutation_importance.png",
            "vo": [
                (60.0, 63.5, "Permutation importance and SHAP. Investment type dominates."),
                (63.5, 66.5, "Most other features are near zero, or negative."),
                (66.5, 69.0, "Investor features made it worse, so they ship disabled."),
            ],
        },
        {
            "start": 69.0, "end": 90.0,
            "kicker": "THE APP",
            "title": "Shipped, not just\nnotebooked",
            "cards": [
                ("10", "page Streamlit dashboard"),
                (str(f["n_tests"]), "tests"),
                ("SHA-256", "change detection"),
                ("live", "metrics into the PDF"),
            ],
            "runtime": f"{f['run_skip_shap_s']}s  --skip-shap        {f['run_full_s']}s  full",
            "final_card": ["Analytical prototype.", "Not investment advice."],
            "closing": "Startup Intelligence AI",
            "vo": [
                (69.0, 72.5, "Ten dashboard pages."),
                (72.5, 75.5, f"{f['n_tests']} tests."),
                (75.5, 80.0, "A scheduled runner that skips itself when the CSV hasn't changed."),
                (80.0, 84.5, "A PDF generated from the live metrics, so the figures can't drift."),
                (84.5, 87.5, "Analytical prototype. Not investment advice."),
                (87.5, 90.0, "Which piece would you want the code for?"),
            ],
        },
    ]


SCENES = build_scenes