import numpy as np
import pandas as pd

from src.preprocessing import (
    clean_text,
    detect_amount_unit_inconsistency,
    normalize_city,
    normalize_industry,
    normalize_investment_type,
    preprocess,
)


class TestCleanText:
    def test_whitespace(self):
        assert clean_text("  hello   world  ") == "hello world"

    def test_nan(self):
        assert clean_text(np.nan) == ""

    def test_none(self):
        assert clean_text(None) == ""


class TestNormalizeCity:
    def test_bangalore_to_bengaluru(self):
        assert normalize_city("Bangalore") == "Bengaluru"

    def test_bengaluru(self):
        assert normalize_city("Bengaluru") == "Bengaluru"

    def test_gurgaon_to_gurugram(self):
        assert normalize_city("Gurgaon") == "Gurugram"

    def test_empty(self):
        assert normalize_city("") == "Unknown"

    def test_nan(self):
        assert normalize_city(np.nan) == "Unknown"

    def test_unmapped_city(self):
        assert normalize_city("Mumbai") == "Mumbai"


class TestNormalizeIndustry:
    def test_ecommerce(self):
        assert normalize_industry("E-commerce") == "E-Commerce"

    def test_fintech(self):
        assert normalize_industry("FinTech") == "FinTech"

    def test_empty(self):
        assert normalize_industry("") == "Unknown"

    def test_nan(self):
        assert normalize_industry(np.nan) == "Unknown"


    def test_nan_string_is_unknown_city(self):
        assert normalize_city("Nan") == "Unknown"

    def test_nan_string_is_unknown_industry(self):
        assert normalize_industry("Nan") == "Unknown"
        assert normalize_industry("N/A") == "Unknown"


class TestNormalizeInvestmentType:
    def test_seed(self):
        assert normalize_investment_type("Seed Round") == "Seed"

    def test_private_equity(self):
        assert normalize_investment_type("Private Equity Round") == "Private Equity"

    def test_empty(self):
        assert normalize_investment_type("") == "Unknown"

    def test_seed_angel_variants_merged(self):
        assert normalize_investment_type("Seed/ Angel Funding") == "Seed/Angel"
        assert normalize_investment_type("Seed / Angel Funding") == "Seed/Angel"
        assert normalize_investment_type("Seed/Angel Funding") == "Seed/Angel"

    def test_literal_backslash_n_cleaned(self):
        assert normalize_investment_type("Seed\\nFunding") == "Seed"
        assert normalize_investment_type("Seed\\\\nFunding") == "Seed"


class TestDetectAmountUnitInconsistency:
    def test_consistent(self):
        df = pd.DataFrame({"amount_in_usd": ["1000000", "2000000", "3000000"]})
        result = detect_amount_unit_inconsistency(df)
        assert result["unit_consistent"] is True


class TestPreprocess:
    def test_basic_preprocessing(self):
        df = pd.DataFrame({
            "Sr No": [1],
            "Date dd/mm/yyyy": ["09/01/2020"],
            "Startup Name": ["TestCo"],
            "Industry Vertical": ["E-commerce"],
            "SubVertical": ["Online marketplace"],
            "City  Location": ["Bangalore"],
            "Investors Name": ["Investor A"],
            "InvestmentnType": ["Seed Round"],
            "Amount in USD": ["10,00,000"],
            "Remarks": [""],
        })
        result = preprocess(df)
        assert len(result) == 1
        assert result.iloc[0]["city_clean"] == "Bengaluru"
        assert result.iloc[0]["industry_clean"] == "E-Commerce"
        assert result.iloc[0]["investment_type_clean"] == "Seed"
        assert result.iloc[0]["amount_clean"] == 1000000.0
        assert result.iloc[0]["year"] == 2020
        assert result.iloc[0]["month"] == 1

    def test_invalid_date_flagged(self):
        df = pd.DataFrame({
            "Sr No": [1],
            "Date dd/mm/yyyy": ["01/07/015"],
            "Startup Name": ["TestCo"],
            "Industry Vertical": ["E-commerce"],
            "SubVertical": ["Online marketplace"],
            "City  Location": ["Bangalore"],
            "Investors Name": ["Investor A"],
            "InvestmentnType": ["Seed Round"],
            "Amount in USD": ["10,00,000"],
            "Remarks": [""],
        })
        result = preprocess(df)
        assert result.iloc[0]["has_valid_date"] == False
        assert result.iloc[0]["date_parsed"] == ""

    def test_undisclosed_amount(self):
        df = pd.DataFrame({
            "Sr No": [1],
            "Date dd/mm/yyyy": ["09/01/2020"],
            "Startup Name": ["TestCo"],
            "Industry Vertical": ["E-commerce"],
            "SubVertical": ["Online marketplace"],
            "City  Location": ["Bangalore"],
            "Investors Name": ["Investor A"],
            "InvestmentnType": ["Seed Round"],
            "Amount in USD": ["undisclosed"],
            "Remarks": [""],
        })
        result = preprocess(df)
        assert np.isnan(result.iloc[0]["amount_clean"])
        assert result.iloc[0]["has_valid_amount"] == False

    def test_duplicate_detection(self):
        df = pd.DataFrame({
            "Sr No": [1, 1],
            "Date dd/mm/yyyy": ["09/01/2020", "09/01/2020"],
            "Startup Name": ["TestCo", "TestCo"],
            "Industry Vertical": ["E-commerce", "E-commerce"],
            "SubVertical": ["Online marketplace", "Online marketplace"],
            "City  Location": ["Bangalore", "Bangalore"],
            "Investors Name": ["Investor A", "Investor A"],
            "InvestmentnType": ["Seed Round", "Seed Round"],
            "Amount in USD": ["10,00,000", "10,00,000"],
            "Remarks": ["", ""],
        })
        result = preprocess(df)
        assert result.iloc[0]["is_duplicate"] == True
        assert result.iloc[1]["is_duplicate"] == True
