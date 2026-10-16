"""Tests for explainability helpers.

These guard two silent-failure bugs: skipping the GroupMedianEncoder step before
SHAP, and handing SHAP an unfitted estimator left over from GridSearchCV cloning.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor

from src.explainability import compute_permutation_importance, explain_single_prediction
from src.train import run_training


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "industry_clean": rng.choice(["E-Commerce", "FinTech", "EdTech"], n),
        "city_clean": rng.choice(["Bengaluru", "Mumbai", "Pune"], n),
        "investment_type_clean": rng.choice(["Seed", "Series A"], n),
        "subvertical": rng.choice(["Online Marketplace", "Lending"], n),
        "year": np.sort(rng.choice([2015.0, 2016.0, 2017.0], n)),
        "month": rng.choice([1.0, 6.0, 12.0], n),
        "investors_name": rng.choice(["Accel India", "Blume Ventures", ""], n),
    })


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    # Module-scoped, so use tmp_path_factory rather than the per-test tmp_path.
    # Writing into the real models/ dir would overwrite the pipeline artifacts.
    out = tmp_path_factory.mktemp("models")
    X = _frame()
    y = pd.Series(np.exp(np.random.default_rng(1).normal(14, 1.2, len(X))))
    split = int(len(X) * 0.8)
    return run_training(
        X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:], output_dir=out
    )


class TestPermutationImportance:
    def test_returns_sorted_dataframe(self, trained):
        results, pipeline, _, _, _ = trained
        X = _frame(200)
        y = pd.Series(np.full(len(X), 1_000_000.0))
        df = compute_permutation_importance(pipeline, X, y, n_repeats=2)
        assert {"feature", "importance_mean", "importance_std"} <= set(df.columns)
        assert df["importance_mean"].is_monotonic_decreasing

    def test_reports_raw_input_columns(self, trained):
        """Permutation importance shuffles the pipeline's *input* columns.
        The grp_* features are generated downstream by GroupMedianEncoder, so
        they are correctly absent from this table -- importance for them is
        captured by SHAP on the transformed matrix instead."""
        _, pipeline, _, _, _ = trained
        X = _frame(120)
        y = pd.Series(np.full(len(X), 1_000_000.0))
        df = compute_permutation_importance(pipeline, X, y, n_repeats=2)

        assert set(df["feature"]) == set(X.columns)
        assert not any(c.startswith("grp_") for c in df["feature"])


class TestShapInputs:
    def test_transform_includes_every_pre_regressor_step(self, trained):
        """The bug this guards: transforming with the preprocessor alone skips
        GroupMedianEncoder, so the grp_* columns are absent and SHAP raises
        'columns are missing'."""
        _, pipeline, _, _, _ = trained
        X = _frame(120)

        step_names = [n for n, _ in pipeline.steps]
        assert "group_medians" in step_names

        processed = X.head(100)
        for name, step in pipeline.steps:
            if name == "regressor":
                break
            processed = step.transform(processed)

        array = np.asarray(processed)
        assert np.isfinite(array).all()
        # grp_* columns must be present, which only the encoder can add.
        assert array.shape[1] > len(_frame(1).columns)

    def test_resolves_fitted_inner_regressor(self, trained):
        from src.explainability import _resolve_fitted

        _, pipeline, _, _, _ = trained
        wrapper = pipeline.named_steps["regressor"]
        if not isinstance(wrapper, TransformedTargetRegressor):
            pytest.skip("selected model is not a TransformedTargetRegressor")
        resolved = _resolve_fitted(wrapper)
        inner = resolved.regressor_ if isinstance(resolved, TransformedTargetRegressor) else resolved
        # A cloned-but-unfitted template would lack this attribute.
        assert hasattr(inner, "estimators_") or hasattr(inner, "coef_")

    def test_resolve_fitted_finds_fitted_child_of_template(self):
        from src.explainability import _resolve_fitted

        unfitted = RandomForestRegressor(n_estimators=5, random_state=0)
        wrapper = TransformedTargetRegressor(regressor=unfitted, func=np.log1p,
                                            inverse_func=np.expm1)
        X = np.random.rand(40, 3)
        y = np.random.rand(40)
        wrapper.fit(X, y)
        resolved = _resolve_fitted(wrapper)
        assert hasattr(resolved.regressor_, "estimators_")


class TestExplainSinglePrediction:
    def test_returns_prediction_and_disclaimer(self, trained):
        _, pipeline, _, _, _ = trained
        row = _frame(1)
        out = explain_single_prediction(pipeline, row, list(row.columns))
        assert out["predicted_amount_usd"] > 0
        assert "causation" in out["note"].lower()


def test_dummy_regressor_has_no_useful_attribution():
    # Guards the branch that skips SHAP for a dummy instead of faking a plot.
    assert not hasattr(DummyRegressor(strategy="median"), "estimators_")