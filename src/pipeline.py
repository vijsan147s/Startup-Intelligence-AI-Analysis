"""Single canonical pipeline: clean -> database -> features -> models -> explain.

Both ``run_pipeline.py`` and ``run_full.py`` are thin entry points over this
module. They previously duplicated each other, which meant a change to one
(adding clustering, say) could silently leave the other producing an incomplete
set of artifacts.
"""

import sys

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor

from .data_loader import load_raw_csv
from .database import create_database, get_db_summary
from .evaluate import plot_actual_vs_predicted, plot_model_comparison, plot_residuals
from .explainability import (
    compute_permutation_importance,
    plot_permutation_importance,
    try_shap_explanation,
)
from .features import create_time_split, prepare_features
from .preprocessing import preprocess, save_processed
from .train import run_training

PERMUTATION_PATH = "data/processed/permutation_importance.csv"


def _log_cv(res: dict, cv_key: str, test_key: str) -> str:
    cv = res.get(cv_key)
    cv_str = f"{cv:.4f}" if cv is not None else "   n/a"
    return f"CV={cv_str}, test={res[test_key]}"


def run(skip_shap: bool = False, skip_classification: bool = False,
        skip_clustering: bool = False, verbose: bool = True) -> dict:

    def say(msg=""):
        if verbose:
            print(msg, flush=True)

    say("=== Step 1: Data Cleaning ===")
    df = preprocess(load_raw_csv())
    save_processed(df)
    say(f"Cleaned data: {len(df)} rows")

    say("\n=== Step 2: Database ===")
    db_path = create_database(df)
    summary = get_db_summary(db_path)
    say(f"Database summary: {summary}")

    say("\n=== Step 3: Feature Preparation ===")
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = create_time_split(X, y)
    say(f"Train: {len(X_train)}, Test: {len(X_test)}")

    say("\n=== Step 4: Regression Models ===")
    results, pipeline, preprocessor, best_name, paths = run_training(
        X_train, y_train, X_test, y_test
    )
    say(f"Best model (by inner CV RMSLE): {best_name}")
    for name, res in results.items():
        say(f"  {name}: {_log_cv(res, 'cv_rmsle', 'rmsle')}, "
            f"MAE={res['mae']:,.0f}, R2={res['r2']:.4f}")
    say(f"  tuned params: {results[best_name].get('best_params')}")

    say("\n=== Step 5: Evaluation Plots ===")
    plot_model_comparison(results)
    best_pred = results[best_name]["predictions"]
    plot_actual_vs_predicted(np.asarray(y_test, dtype=float), best_pred, best_name)
    plot_residuals(np.asarray(y_test, dtype=float), best_pred, best_name)
    say("Model comparison, actual-vs-predicted and residual plots saved")

    say("\n=== Step 6: Permutation Importance ===")
    importance_df = compute_permutation_importance(pipeline, X_test, y_test, n_repeats=10)
    importance_df.to_csv(PERMUTATION_PATH, index=False)
    plot_permutation_importance(importance_df)
    say(f"Top 5 features:\n{importance_df.head().to_string(index=False)}")

    shap_path = None
    if skip_shap:
        say("\n=== Step 7: SHAP (skipped) ===")
    else:
        say("\n=== Step 7: SHAP ===")
        # Explain the model the dashboard actually serves. A DummyRegressor has
        # no meaningful attributions, so SHAP is skipped rather than faked.
        regressor = pipeline.named_steps["regressor"]
        inner = getattr(regressor, "regressor", regressor)
        explain_model = None if isinstance(inner, DummyRegressor) else regressor
        if explain_model is None:
            say("Best model is a dummy; skipping SHAP")
        else:
            shap_path = try_shap_explanation(pipeline, X_test, model=explain_model)
            say(f"SHAP result: {shap_path}")

    clf_results = clf_best = None
    if not skip_classification:
        say("\n=== Step 8: Funding-Range Classification ===")
        from .classification import run_classification
        clf_results, clf_pipeline, clf_best, thresholds, labels, clf_paths = run_classification(
            X_train, y_train, X_test, y_test
        )
        for name, res in clf_results.items():
            say(f"  {name}: {_log_cv(res, 'cv_f1', 'f1_weighted')}")
        say(f"  Best (by inner CV): {clf_best}")
        say(f"  tuned params: {clf_results[clf_best].get('best_params')}")

    clustering = None
    if not skip_clustering:
        say("\n=== Step 9: Clustering ===")
        from .clustering import run_clustering
        clustering = run_clustering(df)
        say(f"k={clustering['optimal_k']} over "
            f"{int((df['has_valid_amount'].astype(str) == 'True').sum())} events")
        cols = ['cluster', 'event_count', 'median_amount', 'avg_year',
                'top_industry', 'top_city']
        say(clustering['summary'][cols].to_string(index=False))

    say("\n=== Pipeline Complete ===")
    return {
        "db_path": db_path,
        "db_summary": summary,
        "results": results,
        "best_name": best_name,
        "pipeline": pipeline,
        "importance": importance_df,
        "shap_path": shap_path,
        "classification_results": clf_results,
        "classification_best": clf_best,
        "clustering": clustering,
    }