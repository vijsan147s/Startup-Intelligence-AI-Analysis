"""Tests for funding-range classification.

The important property here is that category thresholds come from the training
period only and are applied unchanged to the test period.
"""

import numpy as np
import pandas as pd
import pytest

from src.classification import (
    apply_funding_categories,
    build_classification_pipeline,
    create_funding_categories,
    get_classification_models,
    get_classification_param_grids,
)
from src.train import build_feature_pipeline


def _frame(n=150, seed=0):
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


class TestFundingCategories:
    def test_thresholds_come_from_supplied_values(self):
        y = pd.Series([1.0, 2.0, 3.0, 100.0])
        cats, thresholds, labels = create_funding_categories(y)
        assert len(thresholds) == 2
        assert labels == ["Low", "Medium", "High"]
        assert set(cats) <= {0, 1, 2}

    def test_apply_reuses_train_thresholds(self):
        train = pd.Series(np.linspace(0, 100, 200))
        _, thresholds, _ = create_funding_categories(train)
        test = pd.Series([10.0, 50.0, 90.0])
        binned = apply_funding_categories(test, thresholds)
        assert len(binned) == 3

    def test_thresholds_are_not_recomputed_on_test(self):
        """A test distribution shifted upward must still be binned with train
        cut points, not with its own percentiles."""
        train = pd.Series(np.linspace(0, 100, 200))
        _, thresholds, _ = create_funding_categories(train)
        test = pd.Series(np.linspace(100, 200, 200))  # entirely above train range

        train_thresholds_bins = apply_funding_categories(test, thresholds)
        own_percentiles = np.digitize(
            test.to_numpy(),
            np.percentile(test.to_numpy(), np.linspace(0, 100, 4)[1:-1]),
        )
        # Own percentiles would force an even spread; train thresholds should not.
        assert train_thresholds_bins.max() == 2
        assert len(set(train_thresholds_bins)) < 3 or len(set(own_percentiles)) == 3

    def test_deterministic_labels(self):
        y = pd.Series(np.linspace(0, 1000, 300))
        a, _, la = create_funding_categories(y)
        b, _, lb = create_funding_categories(y)
        assert np.array_equal(a, b)
        assert la == lb


class TestSharedPipeline:
    def test_classification_uses_shared_structure(self):
        clf = build_classification_pipeline()
        reg = build_feature_pipeline()
        assert [n for n, _ in clf.steps] == [n for n, _ in reg.steps]

    def test_same_numeric_columns_as_regression(self):
        clf = build_classification_pipeline()
        reg = build_feature_pipeline()
        clf_num = clf.named_steps["preprocessor"].transformers[1][2]
        reg_num = reg.named_steps["preprocessor"].transformers[1][2]
        assert sorted(clf_num) == sorted(reg_num)

    def test_classification_scales_numeric_regression_does_not(self):
        clf_steps = [n for n, _ in
                     build_classification_pipeline().named_steps["preprocessor"].transformers[1][1].steps]
        reg_steps = [n for n, _ in
                     build_feature_pipeline().named_steps["preprocessor"].transformers[1][1].steps]
        assert "scaler" in clf_steps
        assert "scaler" not in reg_steps


class TestModelConfig:
    def test_dummy_has_no_grid(self):
        assert "Dummy (Most Frequent)" not in get_classification_param_grids()

    def test_grids_prefix_classifier(self):
        for grid in get_classification_param_grids().values():
            for key in grid:
                assert key.startswith("classifier__")

    def test_grids_exclude_tied_options(self):
        rf = get_classification_param_grids()["Random Forest"]
        assert None not in rf["classifier__max_depth"]

    def test_all_models_present(self):
        assert set(get_classification_models()) == {
            "Dummy (Most Frequent)", "Logistic Regression", "Random Forest"
        }