import numpy as np
import pandas as pd
import pytest

from src.features import create_time_split, prepare_features
from src.predict import FundingPredictor
from src.preprocessing import preprocess
from src.train import run_training


@pytest.fixture
def sample_processed_df():
    data = {
        "Sr No": list(range(100)),
        "Date dd/mm/yyyy": [f"{1 + i % 28:02d}/{1 + i % 12:02d}/{2015 + i % 5}" for i in range(100)],
        "Startup Name": [f"Startup{i}" for i in range(100)],
        "Industry Vertical": [["E-commerce", "FinTech", "Technology", "SaaS"][i % 4] for i in range(100)],
        "SubVertical": [["Marketplace", "Payments", "Cloud", "CRM"][i % 4] for i in range(100)],
        "City  Location": [["Bangalore", "Mumbai", "Gurgaon", "Pune"][i % 4] for i in range(100)],
        "Investors Name": [f"Investor{i % 10}" for i in range(100)],
        "InvestmentnType": [["Seed Round", "Series A", "Series B", "Private Equity Round"][i % 4] for i in range(100)],
        "Amount in USD": [f"{100000 * (i + 1):,}" for i in range(100)],
        "Remarks": [""] * 100,
    }
    raw = pd.DataFrame(data)
    return preprocess(raw)


class TestPrepareFeatures:
    def test_output_shapes(self, sample_processed_df):
        X, y = prepare_features(sample_processed_df)
        assert len(X) == len(y)
        assert len(X) > 0

    def test_columns(self, sample_processed_df):
        X, y = prepare_features(sample_processed_df)
        expected = [
            "industry_clean", "subvertical", "city_clean",
            "investment_type_clean", "year", "month", "investors_name",
        ]
        assert list(X.columns) == expected


class TestCreateTimeSplit:
    def test_split(self, sample_processed_df):
        X, y = prepare_features(sample_processed_df)
        X_train, X_test, y_train, y_test = create_time_split(X, y, test_size=0.3)
        assert len(X_train) + len(X_test) == len(X)
        assert len(X_train) > len(X_test)

    def test_train_before_test(self, sample_processed_df):
        X, y = prepare_features(sample_processed_df)
        X_train, X_test, y_train, y_test = create_time_split(X, y)
        assert X_train["year"].max() <= X_test["year"].min()


class TestFundingPredictor:
    def test_predict_valid(self, sample_processed_df, tmp_path):
        X, y = prepare_features(sample_processed_df)
        X_train, X_test, y_train, y_test = create_time_split(X, y)
        results, pipeline, preprocessor, best_name, paths = run_training(
            X_train, y_train, X_test, y_test, output_dir=tmp_path
        )

        predictor = FundingPredictor(paths["pipeline_path"])
        result = predictor.predict("E-commerce", "Marketplace", "Bangalore", "Seed", 2020, 6)
        assert "estimated_amount_usd" in result
        assert result["estimated_amount_usd"] >= 0
        assert "disclaimer" in result

    def test_predict_invalid_year(self, tmp_path):
        df = pd.DataFrame({
            "industry_clean": ["E-commerce"],
            "subvertical": ["Marketplace"],
            "city_clean": ["Bangalore"],
            "investment_type_clean": ["Seed"],
            "year": [2020.0],
            "month": [6.0],
        })
        pipeline_path = tmp_path / "pipeline.joblib"
        from sklearn.pipeline import Pipeline
        from sklearn.dummy import DummyRegressor
        pipe = Pipeline(steps=[("preprocessor", None), ("regressor", DummyRegressor(strategy="median"))])
        import joblib
        joblib.dump(pipe, pipeline_path)

        predictor = FundingPredictor(pipeline_path)
        with pytest.raises(ValueError, match="outside the supported range"):
            predictor.predict("E-commerce", "Marketplace", "Bangalore", "Seed", 1990, 6)

    def test_predict_invalid_month(self):
        from sklearn.pipeline import Pipeline
        from sklearn.dummy import DummyRegressor
        pipe = Pipeline(steps=[("preprocessor", None), ("regressor", DummyRegressor(strategy="median"))])
        import joblib, tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        joblib.dump(pipe, path)
        predictor = FundingPredictor(path)
        with pytest.raises(ValueError, match="Month"):
            predictor.predict("E-commerce", "Marketplace", "Bangalore", "Seed", 2020, 13)
        os.unlink(path)
