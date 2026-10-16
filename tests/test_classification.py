import numpy as np
import pandas as pd
import pytest

from src.classification import create_funding_categories, run_classification
from src.features import create_time_split, prepare_features
from src.preprocessing import preprocess


@pytest.fixture
def sample_data():
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
    df = preprocess(raw)
    X, y = prepare_features(df)
    return create_time_split(X, y)


class TestCreateFundingCategories:
    def test_three_categories(self, sample_data):
        X_train, X_test, y_train, y_test = sample_data
        categories, thresholds, labels = create_funding_categories(y_train)
        assert len(labels) == 3
        assert len(thresholds) == 2
        assert set(categories).issubset({0, 1, 2})


class TestRunClassification:
    def test_classification_runs(self, sample_data, tmp_path):
        X_train, X_test, y_train, y_test = sample_data
        results, pipeline, best_name, thresholds, labels, paths = run_classification(
            X_train, y_train, X_test, y_test, output_dir=tmp_path
        )
        assert best_name in results
        assert "f1_weighted" in results[best_name]
        assert len(paths) == 2
