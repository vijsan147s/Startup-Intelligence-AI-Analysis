"""Tests for GroupMedianEncoder and InvestorFeatureBuilder.

The property that matters most here is leak safety: an encoder fitted on the
training period must never reveal target information from rows it has not seen.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    GroupMedianEncoder,
    InvestorFeatureBuilder,
    lead_investor_key,
    parse_investor_names,
)


def _frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "industry_clean": rng.choice(["E-Commerce", "FinTech", "EdTech"], n),
        "city_clean": rng.choice(["Bengaluru", "Mumbai", "Pune"], n),
        "investment_type_clean": rng.choice(["Seed", "Series A", "Debt Funding"], n),
        "subvertical": rng.choice(["Online Marketplace", "Lending"], n),
        "year": rng.choice([2015.0, 2016.0, 2017.0], n),
        "month": rng.choice([1.0, 6.0, 12.0], n),
        "investors_name": rng.choice(
            ["Accel India, Tiger Global", "Blume Ventures", "", "undisclosed"], n
        ),
    })


class TestGroupMedianEncoder:
    def test_output_columns_created(self):
        X = _frame()
        y = pd.Series(np.arange(len(X), dtype=float))
        enc = GroupMedianEncoder().fit(X, y)
        out = enc.transform(X)
        for spec in GroupMedianEncoder().specs:
            assert f"grp_{spec}" in out.columns

    def test_transform_does_not_mutate_input(self):
        X = _frame()
        y = pd.Series(np.arange(len(X), dtype=float))
        before = X.copy()
        GroupMedianEncoder().fit(X, y).transform(X)
        pd.testing.assert_frame_equal(X, before)

    def test_no_nan_in_output(self):
        X = _frame()
        y = pd.Series(np.arange(len(X), dtype=float))
        out = GroupMedianEncoder().fit(X, y).transform(X)
        grp_cols = [c for c in out.columns if c.startswith("grp_")]
        assert out[grp_cols].notna().all().all()

    def test_unseen_groups_fall_back_to_global_median(self):
        X = _frame()
        y = pd.Series(np.arange(len(X), dtype=float))
        enc = GroupMedianEncoder(specs={"industry": ["industry_clean"]}, min_group_size=1)
        enc.fit(X, y)

        unseen = X.head(3).copy()
        unseen["industry_clean"] = "Totally New Sector"
        out = enc.transform(unseen)
        assert (out["grp_industry"] == enc.global_median_).all()

    def test_small_groups_collapse_to_global_median(self):
        X = _frame(n=30)
        y = pd.Series(np.arange(len(X), dtype=float))
        enc = GroupMedianEncoder(specs={"industry": ["industry_clean"]}, min_group_size=1000)
        enc.fit(X, y)
        # Every group is below the threshold, so none are retained.
        assert enc.medians_.get("industry") == {}

    def test_encode_does_not_see_unseen_rows(self):
        """A group absent from training must not appear in the fitted medians,
        which is the mechanism that prevents target leakage."""
        train = _frame(n=30, seed=1)
        train["industry_clean"] = "E-Commerce"
        y_train = pd.Series(np.full(len(train), 500.0))
        enc = GroupMedianEncoder(specs={"industry": ["industry_clean"]}, min_group_size=1).fit(
            train, y_train
        )
        assert set(enc.medians_["industry"]) == {"E-Commerce"}


class TestParseInvestorNames:
    @pytest.mark.parametrize("value,expected", [
        ("Accel India, Tiger Global", ["accel india", "tiger global"]),
        ("Blume Ventures", ["blume ventures"]),
        ("", []),
        ("undisclosed", []),
        ("Undisclosed", []),
        (None, []),
        (np.nan, []),
        ("A;B&C", ["a", "b", "c"]),
    ])
    def test_parsing(self, value, expected):
        assert parse_investor_names(value) == expected


class TestInvestorFeatureBuilder:
    def test_counts_syndrome_size(self):
        X = pd.DataFrame({"investors_name": ["A, B, C", "A", ""]})
        out = InvestorFeatureBuilder().fit(X).transform(X)
        assert list(out["investor_count"]) == [3.0, 1.0, 0.0]

    def test_flags_top_investors(self):
        X = _frame(n=60, seed=2)
        builder = InvestorFeatureBuilder(min_appearances=1)
        builder.fit(X)
        assert builder.top_investors_
        out = builder.transform(X)
        flag_cols = [c for c in out.columns if c.startswith("top_investor_")]
        assert flag_cols
        assert set(np.unique(out[flag_cols].to_numpy())) <= {0.0, 1.0}

    def test_history_count_zero_for_unknown_investor(self):
        # Fitted on one set of rows, applied to investors absent from it.
        train = pd.DataFrame({"investors_name": ["Accel India", "Blume Ventures"]})
        builder = InvestorFeatureBuilder().fit(train)
        unseen = pd.DataFrame({"investors_name": ["Nobody At All", ""]})
        out = builder.transform(unseen)
        assert (out["investor_history_count"] == 0).all()

    def test_history_count_reflects_training_frequency(self):
        train = pd.DataFrame({"investors_name": ["Accel India"] * 5 + ["Blume Ventures"]})
        builder = InvestorFeatureBuilder().fit(train)
        out = builder.transform(train)
        assert out.loc[0, "investor_history_count"] == 5
        assert out.loc[5, "investor_history_count"] == 1

    def test_handles_missing_column(self):
        X = _frame(n=10).drop(columns=["investors_name"])
        out = InvestorFeatureBuilder().fit(X).transform(X)
        assert (out["investor_count"] == 0).all()


class TestLeadInvestorKey:
    def test_returns_first_investor(self):
        s = pd.Series(["Accel India, Tiger Global", "Blume Ventures", ""])
        assert list(lead_investor_key(s)) == ["accel india", "blume ventures", "Unknown"]


def test_feature_columns_include_investors():
    assert "investors_name" in FEATURE_COLUMNS