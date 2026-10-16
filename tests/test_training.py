"""Tests for model tuning, metrics and the prediction interface."""

import numpy as np
import pandas as pd
import pytest

from src.config import MAX_PREDICTABLE_YEAR, MIN_PREDICTABLE_YEAR, USD_TO_INR
from src.features import FEATURE_COLUMNS
from src.train import (
    evaluate_predictions,
    get_inner_cv_scorer,
    get_param_grids,
    root_mean_squared_log_error,
    tune_and_fit,
)


def _frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "industry_clean": rng.choice(["E-Commerce", "FinTech", "EdTech"], n),
        "city_clean": rng.choice(["Bengaluru", "Mumbai", "Pune"], n),
        "investment_type_clean": rng.choice(["Seed", "Series A", "Debt Funding"], n),
        "subvertical": rng.choice(["Online Marketplace", "Lending"], n),
        "year": np.sort(rng.choice([2015.0, 2016.0, 2017.0], n)),
        "month": rng.choice([1.0, 6.0, 12.0], n),
        "investors_name": rng.choice(["Accel India", "Blume Ventures", ""], n),
    })


class TestMetrics:
    def test_rmsle_zero_for_perfect_predictions(self):
        y = np.array([1.0, 100.0, 10000.0])
        assert root_mean_squared_log_error(y, y) == pytest.approx(0.0)

    def test_rmsle_handles_negative_predictions(self):
        # expm1 of a log prediction can go negative; the metric must not explode.
        assert np.isfinite(root_mean_squared_log_error([10.0, 20.0], [-5.0, 20.0]))

    def test_evaluate_predictions_reports_all_metrics(self):
        out = evaluate_predictions([1.0, 2.0, 3.0], [1.1, 2.1, 2.7])
        for key in ["mae", "rmse", "r2", "rmsle"]:
            assert key in out

    def test_scorer_is_higher_is_better_false(self):
        scorer = get_inner_cv_scorer()
        assert scorer._sign < 0


class TestParamGrids:
    def test_dummy_has_no_grid(self):
        assert "Dummy (Median)" not in get_param_grids()

    def test_grids_target_the_inner_regressor(self):
        for grid in get_param_grids().values():
            for key in grid:
                # regressor__regressor__<param>: outer TTR, inner estimator.
                assert key.startswith("regressor__regressor__")

    def test_grids_exclude_tied_options(self):
        """max_depth=None ties exactly with max_depth=20 on this data, so
        keeping both made the winner depend on parallel search scheduling."""
        rf = get_param_grids()["Random Forest"]
        assert None not in rf["regressor__regressor__max_depth"]

    def test_no_duplicate_values_in_any_grid(self):
        for grid in get_param_grids().values():
            for values in grid.values():
                assert len(values) == len(set(values))


class TestTuneAndFit:
    def test_returns_fitted_pipeline_predicting_dollars(self):
        X = _frame()
        y = pd.Series(np.exp(np.random.default_rng(3).normal(14, 1.5, len(X))))

        from sklearn.linear_model import Ridge
        from sklearn.compose import TransformedTargetRegressor

        wrapped = TransformedTargetRegressor(
            regressor=Ridge(), func=np.log1p, inverse_func=np.expm1
        )
        fitted, params, cv_score = tune_and_fit("Ridge Regression", wrapped, X, y)

        preds = fitted.predict(X)
        # Must be raw dollars, not log-scale: median should sit in the millions.
        assert np.median(preds) > 1e5
        assert any("alpha" in k for k in params)
        assert cv_score > 0

    def test_untuned_model_skips_search(self):
        X = _frame(n=60)
        y = pd.Series(np.full(len(X), 1000.0))

        from sklearn.dummy import DummyRegressor

        fitted, params, cv_score = tune_and_fit(
            "Dummy (Median)", DummyRegressor(strategy="median"), X, y
        )
        assert params == {}
        assert np.isnan(cv_score)
        assert fitted.predict(X).shape == (len(X),)


def test_predict_rejects_out_of_range_year():
    from src.predict import FundingPredictor
    predictor = FundingPredictor.__new__(FundingPredictor)

    with pytest.raises(ValueError, match="supported range"):
        predictor.predict("E-Commerce", "", "Bengaluru", "Seed", MIN_PREDICTABLE_YEAR - 1, 6)


def test_predict_rejects_invalid_month():
    from src.predict import FundingPredictor
    predictor = FundingPredictor.__new__(FundingPredictor)

    with pytest.raises(ValueError, match="between 1 and 12"):
        predictor.predict("E-Commerce", "", "Bengaluru", "Seed", 2018, 13)


def test_feature_columns_are_passthrough_safe():
    # predict.py builds a row from exactly these keys.
    required = set(FEATURE_COLUMNS)
    assert {"industry_clean", "city_clean", "investment_type_clean", "year", "month"} <= required


def test_fx_rate_is_configurable():
    assert USD_TO_INR > 0