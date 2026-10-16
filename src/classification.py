"""Funding-range classification (Low / Medium / High).

Feature construction is delegated to the shared builders in ``src/train.py`` so
regression and classification consume an identical column set. Enabling investor
features or changing the group-median specs therefore affects both models at
once, instead of the two drifting apart.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from .config import CV_N_SPLITS, MODELS_DIR, RANDOM_STATE
from .train import INCLUDE_INVESTOR_FEATURES, build_feature_pipeline

logger = logging.getLogger(__name__)


def create_funding_categories(y_train, n_categories: int = 3):
    """Derive Low/Medium/High cut points from training amounts only.

    Thresholds computed on the test period would leak distributional
    information across the temporal split.
    """
    thresholds = np.percentile(y_train, np.linspace(0, 100, n_categories + 1)[1:-1])
    categories = np.digitize(y_train, thresholds)
    labels = ["Low", "Medium", "High"][:n_categories]
    return categories, thresholds, labels


def apply_funding_categories(y, thresholds) -> np.ndarray:
    """Bin new amounts using thresholds fixed on the training period."""
    return np.digitize(np.asarray(y, dtype=float), thresholds)


def build_classification_pipeline() -> Pipeline:
    """Shared feature pipeline with numeric scaling enabled for classifiers."""
    return build_feature_pipeline(
        include_investors=INCLUDE_INVESTOR_FEATURES,
        scale_numeric=True,
    )


def get_classification_models() -> dict:
    return {
        "Dummy (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def get_classification_param_grids() -> dict:
    return {
        "Logistic Regression": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__max_iter": [1000],
        },
        "Random Forest": {
            # max_depth=None ties exactly with max_depth=20 here; keeping only
            # 20 makes the selection reproducible rather than tie-break dependent.
            "classifier__max_depth": [8, 20],
            "classifier__min_samples_leaf": [5, 20],
            "classifier__n_estimators": [200],
        },
    }


def train_classification_models(X_train, y_train_cat, X_test, y_test_cat, labels):
    grids = get_classification_param_grids()
    results = {}

    for name, model in get_classification_models().items():
        logger.info("Training %s", name)

        pipeline = build_classification_pipeline()
        pipeline.steps.append(("classifier", model))

        if name in grids:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=grids[name],
                scoring="f1_weighted",
                # TimeSeriesSplit: rows are year-ordered, so a random fold
                # would train on later years and validate on earlier ones.
                cv=TimeSeriesSplit(n_splits=CV_N_SPLITS),
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train_cat)
            fitted = search.best_estimator_
            best_params = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
            cv_score = float(search.best_score_)
            logger.info("%s - best inner CV F1 %.4f with %s", name, cv_score, best_params)
        else:
            fitted = pipeline.fit(X_train, y_train_cat)
            best_params, cv_score = {}, float("nan")

        y_pred = fitted.predict(X_test)

        results[name] = {
            "model": fitted,
            "f1_weighted": round(float(f1_score(y_test_cat, y_pred, average="weighted")), 4),
            "cv_f1": None if np.isnan(cv_score) else round(cv_score, 4),
            "best_params": best_params,
            "report": classification_report(y_test_cat, y_pred, target_names=labels, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test_cat, y_pred).tolist(),
            "predictions": y_pred,
        }
        logger.info("%s - test F1 (weighted): %.4f", name, results[name]["f1_weighted"])

    # Selected by inner CV, not by test F1, for the same reason the regression
    # task selects on CV: choosing on the test period invalidates that number.
    scored = {k: v for k, v in results.items() if v["cv_f1"] is not None}
    best_name = max(scored, key=lambda k: scored[k]["cv_f1"])
    logger.info("Best classification model (by inner CV): %s", best_name)

    return results, results[best_name]["model"], best_name


def save_classification_pipeline(pipeline, results, best_name, thresholds, labels, output_dir=None):
    output_dir = Path(output_dir) if output_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = output_dir / "classification_pipeline.joblib"
    metrics_path = output_dir / "classification_metrics.json"

    joblib.dump(pipeline, pipeline_path)

    serializable = {}
    for name, res in results.items():
        serializable[name] = {
            "f1_weighted": res["f1_weighted"],
            "cv_f1": res.get("cv_f1"),
            "best_params": res.get("best_params", {}),
            "report": res["report"],
            "confusion_matrix": res["confusion_matrix"],
        }
    serializable["best_model"] = best_name
    serializable["selection_metric"] = "cv_f1"
    serializable["thresholds"] = np.asarray(thresholds).tolist()
    serializable["labels"] = labels

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    logger.info("Classification pipeline saved to %s", pipeline_path)
    return str(pipeline_path), str(metrics_path)


def run_classification(X_train, y_train, X_test, y_test, output_dir=None):
    y_train_cat, thresholds, labels = create_funding_categories(y_train)
    # Reuse the training thresholds instead of recomputing them on the test set.
    y_test_cat = apply_funding_categories(y_test, thresholds)

    results, pipeline, best_name = train_classification_models(
        X_train, y_train_cat, X_test, y_test_cat, labels
    )
    paths = save_classification_pipeline(
        pipeline, results, best_name, thresholds, labels, output_dir=output_dir
    )

    return results, pipeline, best_name, thresholds, labels, paths