import logging
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "industry_clean",
    "subvertical",
    "city_clean",
    "investment_type_clean",
    "year",
    "month",
    "investors_name",
]
TARGET_COLUMN = "amount_clean"

# Groups coarse enough to have a reliable median. `subvertical` is deliberately
# excluded: it has ~1,300 distinct values across ~2,000 rows, so any per-group
# median degenerates into a memorised one-off and hurts on the test period.
GROUP_MEDIAN_SPECS = {
    "industry_year": ["industry_clean", "year"],
    "city_year": ["city_clean", "year"],
    "industry": ["industry_clean"],
    "city": ["city_clean"],
    "investment_type": ["investment_type_clean"],
    "industry_city": ["industry_clean", "city_clean"],
}

# Groups smaller than this fall back to the global training median. Tuned on the
# time-based split: 10 beat 1, 3, 20 and 25.
MIN_GROUP_SIZE = 10


def lead_investor_key(series: pd.Series) -> pd.Series:
    """Collapse a syndicate to its first-listed investor, for group-median use."""
    return series.apply(lambda v: (parse_investor_names(v) or ["Unknown"])[0])


def _group_key(df: pd.DataFrame, cols: list) -> pd.Series:
    return df[cols].astype(str).agg("|".join, axis=1)


def parse_investor_names(value) -> list:
    """Split a raw ``investors_name`` cell into normalised investor names.

    The source column packs syndicates into one string, e.g.
    ``"Accel India, Tiger Global"``. Undisclosed and placeholder values yield
    an empty list rather than a fake investor.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in MISSING_INVESTOR_TOKENS:
        return []
    parts = re.split(r"[,;&|]", text)
    names = [p.strip().lower() for p in parts if p.strip()]
    return [n for n in names if n not in MISSING_INVESTOR_TOKENS]


MISSING_INVESTOR_TOKENS = {
    "", "nan", "none", "undisclosed", "unknown", "not disclosed", "not available",
    "-", "--", "na", "n/a", "tbd",
}

# Investors seen often enough in the training period to be worth a flag.
MIN_INVESTOR_APPEARANCES = 10


class InvestorFeatureBuilder(BaseEstimator, TransformerMixin):
    """Derives syndicate-shape features from ``investors_name``.

    Produces three leak-safe signals:

    * ``investor_count`` - how many investors backed the round
    * ``top_investor_*`` - one binary flag per frequently-seen investor,
      identified during ``fit`` on training rows only
    * ``investor_history_count`` - how often the round's lead investor had
      already appeared earlier in the training period

    Note on availability: investor identity is knowable at deal time in practice,
    but not from this dataset alone for a brand-new startup. Treat these features
    as describing historical syndicate patterns, not as a forward-looking signal.
    """

    def __init__(self, min_appearances: int = MIN_INVESTOR_APPEARANCES):
        self.min_appearances = min_appearances

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        self.investor_counts_ = (
            X["investors_name"].apply(parse_investor_names).explode().value_counts()
            if "investors_name" in X.columns else pd.Series(dtype=int)
        )
        self.top_investors_ = sorted(
            self.investor_counts_[self.investor_counts_ >= self.min_appearances].index.tolist()
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        parsed = (
            X["investors_name"].apply(parse_investor_names)
            if "investors_name" in X.columns
            else X.apply(lambda _: [], axis=1)
        )

        X["investor_count"] = parsed.apply(len).astype(float)
        X["has_investors"] = (X["investor_count"] > 0).astype(float)
        for investor in getattr(self, "top_investors_", []):
            X[f"top_investor_{investor}"] = parsed.apply(
                lambda names, inv=investor: float(inv in names)
            )

        # Longest-standing investor's training-period frequency, as a proxy for
        # how seasoned the syndicate is.
        counts = getattr(self, "investor_counts_", pd.Series(dtype=int))
        X["investor_history_count"] = parsed.apply(
            lambda names: float(max([counts.get(n, 0) for n in names], default=0))
        )
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else list(FEATURE_COLUMNS)
        extra = ["investor_count", "has_investors", "investor_history_count"]
        extra += [f"top_investor_{i}" for i in getattr(self, "top_investors_", [])]
        return base + extra


class GroupMedianEncoder(BaseEstimator, TransformerMixin):
    """Adds median-target-encoded columns for coarse categorical groups.

    Medians are computed during ``fit`` on training rows only, so no target
    information from the validation period can leak into the features. At
    transform time unseen groups fall back to the global training median.
    """

    def __init__(self, specs: dict | None = None, min_group_size: int = MIN_GROUP_SIZE):
        self.specs = specs if specs is not None else GROUP_MEDIAN_SPECS
        self.min_group_size = min_group_size

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        y = pd.Series(np.asarray(y), index=X.index)
        self.global_median_ = float(y.median())
        self.medians_: dict = {}
        for name, cols in self.specs.items():
            if not all(c in X.columns for c in cols):
                continue
            grouped = y.groupby(_group_key(X, cols)).agg(["median", "size"])
            # Groups too small to be reliable collapse to the global median.
            grouped = grouped[grouped["size"] >= self.min_group_size]
            self.medians_[name] = grouped["median"].to_dict()
        self.feature_names_out_ = list(self.specs.keys())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for name, cols in self.specs.items():
            col = f"grp_{name}"
            mapping = getattr(self, "medians_", {}).get(name)
            if mapping is None or not all(c in X.columns for c in cols):
                X[col] = getattr(self, "global_median_", np.nan)
                continue
            X[col] = _group_key(X, cols).map(mapping).fillna(self.global_median_)
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else list(FEATURE_COLUMNS)
        return base + [f"grp_{name}" for name in self.specs]


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = df[df["has_valid_amount"]].copy()
    df = df[df["has_valid_date"]].copy()

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X["year"] = X["year"].fillna(X["year"].median())
    X["month"] = X["month"].fillna(X["month"].median())

    for col in ["industry_clean", "subvertical", "city_clean", "investment_type_clean"]:
        X[col] = X[col].fillna("Unknown")
        X[col] = X[col].astype(str)

    if "investors_name" not in X.columns:
        X["investors_name"] = ""
    X["investors_name"] = X["investors_name"].fillna("").astype(str)

    logger.info("Features prepared: %d samples, %d columns", len(X), X.shape[1])
    return X, y


def create_time_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    df = X.copy()
    df["_target"] = y.values
    df = df.sort_values("year")
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["_target"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["_target"]
    logger.info("Time-based split: %d train, %d test", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test
