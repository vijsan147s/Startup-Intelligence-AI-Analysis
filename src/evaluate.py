import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import REPORTS_DIR

logger = logging.getLogger(__name__)


def plot_actual_vs_predicted(y_true, y_pred, model_name: str, output_path=None) -> Path:
    output_path = Path(output_path) if output_path else REPORTS_DIR / f"actual_vs_predicted_{model_name.replace(' ', '_').lower()}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect prediction")
    ax.set_xlabel("Actual Funding (USD)")
    ax.set_ylabel("Predicted Funding (USD)")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.legend()
    ax.set_xlim(0, np.percentile(y_true, 99))
    ax.set_ylim(0, np.percentile(y_pred, 99))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved actual vs predicted plot to %s", output_path)
    return output_path


def plot_residuals(y_true, y_pred, model_name: str, output_path=None) -> Path:
    output_path = Path(output_path) if output_path else REPORTS_DIR / f"residuals_{model_name.replace(' ', '_').lower()}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0].axhline(y=0, color="r", linestyle="--")
    axes[0].set_xlabel("Predicted Funding (USD)")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].set_title(f"Residual Plot — {model_name}")

    axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Residual (USD)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Residual Distribution — {model_name}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved residuals plot to %s", output_path)
    return output_path


def plot_model_comparison(results: dict, output_path=None) -> Path:
    output_path = Path(output_path) if output_path else REPORTS_DIR / "model_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(results.keys())
    maes = [results[n]["mae"] for n in names]
    rmses = [results[n]["rmse"] for n in names]
    r2s = [results[n]["r2"] for n in names]
    rmsles = [results[n].get("rmsle", float("nan")) for n in names]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    x = np.arange(len(names))

    axes[0].bar(x, maes, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].set_ylabel("MAE (USD)")
    axes[0].set_title("Mean Absolute Error")

    axes[1].bar(x, rmses, color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right")
    axes[1].set_ylabel("RMSE (USD)")
    axes[1].set_title("Root Mean Squared Error")

    axes[2].bar(x, r2s, color="seagreen")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha="right")
    axes[2].set_ylabel("R²")
    axes[2].set_title("R-squared")

    axes[3].bar(x, rmsles, color="mediumpurple")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(names, rotation=45, ha="right")
    axes[3].set_ylabel("RMSLE")
    axes[3].set_title("RMSLE (selection metric, lower is better)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved model comparison plot to %s", output_path)
    return output_path


def generate_evaluation_report(results: dict, y_test, best_name: str) -> dict:
    report = {
        "best_model": best_name,
        "selection_metric": "rmsle",
        "evaluation_samples": int(len(y_test)),
        "metrics": {
            name: {
                "mae": r["mae"],
                "rmse": r["rmse"],
                "r2": r["r2"],
                "rmsle": r.get("rmsle"),
            }
            for name, r in results.items()
        },
    }
    return report
