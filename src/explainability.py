import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import REPORTS_DIR
from sklearn.dummy import DummyRegressor
from sklearn.compose import TransformedTargetRegressor

logger = logging.getLogger(__name__)


def compute_permutation_importance(pipeline, X_test, y_test, n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    logger.info("Permutation importance computed for %d features", len(importance_df))
    return importance_df


def plot_permutation_importance(importance_df: pd.DataFrame, output_path=None, top_n: int = 20) -> Path:
    output_path = Path(output_path) if output_path else REPORTS_DIR / "permutation_importance.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["importance_mean"], xerr=df["importance_std"], color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance (increase in MAE when feature is permuted)")
    ax.set_title(f"Top {top_n} Permutation Importance Features")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved permutation importance plot to %s", output_path)
    return output_path


def _resolve_fitted(model):
    """Return a fitted instance of ``model``.

    A hyperparameter search clones candidates, so the wrapper stored in a saved
    pipeline may still reference an unfitted inner estimator. Preferring the
    fitted attributes avoids passing an unfitted model to SHAP.
    """
    fitted_attr = "regressor_" if isinstance(model, TransformedTargetRegressor) else None
    if fitted_attr and getattr(model, fitted_attr, None) is not None:
        return model
    if not hasattr(model, "estimators_") and not hasattr(model, "coef_"):
        for _, candidate in model.get_params(deep=True).items():
            if hasattr(candidate, "estimators_") or hasattr(candidate, "coef_"):
                return candidate
    return model


def try_shap_explanation(pipeline, X_sample, output_path=None, model=None) -> Path | None:
    try:
        import shap
        if model is None:
            model = pipeline.named_steps["regressor"]
        # GridSearchCV clones estimators while searching, so the regressor held
        # by the saved pipeline can be an unfitted template. Always resolve the
        # fitted instance out of the estimator's own parameters.
        model = _resolve_fitted(model)

        # Unwrap TransformedTargetRegressor: it is fitted in log space, so SHAP
        # must explain the inner regressor against log-space targets.
        is_log_space = isinstance(model, TransformedTargetRegressor)
        if is_log_space:
            model = model.regressor_

        # Every step before "regressor" must be applied in order. Calling the
        # preprocessor alone skips GroupMedianEncoder, which is what creates the
        # grp_* columns the preprocessor then expects.
        X_processed = X_sample[:100]
        for step_name, step in pipeline.steps:
            if step_name == "regressor":
                break
            X_processed = step.transform(X_processed)

        if hasattr(model, "predict") and not isinstance(model, DummyRegressor):
            explainer = shap.Explainer(model, X_processed)
            shap_values = explainer(X_processed)

            output_path = Path(output_path) if output_path else REPORTS_DIR / "shap_summary.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_processed, show=False)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            if is_log_space:
                logger.info(
                    "SHAP values are in log1p space; feature direction on the plot "
                    "refers to effect on log amount, not raw USD."
                )
            logger.info("SHAP summary plot saved to %s", output_path)
            return output_path
        else:
            logger.warning("Model is not compatible with SHAP. Skipping SHAP analysis.")
            return None
    except Exception as e:
        logger.warning("SHAP explanation failed: %s. Falling back to permutation importance only.", e)
        return None


def explain_single_prediction(pipeline, input_df: pd.DataFrame, feature_names: list[str]) -> dict:
    prediction = pipeline.predict(input_df)[0]
    explanation = {
        "predicted_amount_usd": round(float(prediction), 2),
        "input_features": input_df.iloc[0].to_dict(),
        "note": (
            "This estimate is based on patterns learned from historical funding data (2015-2020). "
            "Feature associations do not imply causation."
        ),
    }
    return explanation
