import pandas as pd
import pytest

from src.data_loader import (
    EXPECTED_COLUMNS,
    load_raw_csv,
    normalize_column_names,
    parse_amount,
    parse_date,
    validate_schema,
)


class TestLoadRawCsv:
    def test_loads_csv(self):
        df = load_raw_csv()
        assert len(df) > 0
        assert len(df.columns) == 10

    def test_expected_columns(self):
        df = load_raw_csv()
        for col in EXPECTED_COLUMNS:
            assert col in df.columns

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_raw_csv("nonexistent.csv")


class TestValidateSchema:
    def test_valid_schema(self):
        df = load_raw_csv()
        assert validate_schema(df) == []

    def test_missing_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="Missing expected columns"):
            validate_schema(df)


class TestParseDate:
    def test_valid_date(self):
        result = parse_date("09/01/2020")
        assert result == pd.Timestamp("2020-01-09")

    def test_invalid_date(self):
        result = parse_date("not a date")
        assert pd.isna(result)

    def test_empty_string(self):
        result = parse_date("")
        assert pd.isna(result)

    def test_nan(self):
        result = parse_date(None)
        assert pd.isna(result)

    def test_undisclosed(self):
        result = parse_date("undisclosed")
        assert pd.isna(result)

    def test_out_of_range_year_rejected(self):
        # Typo in source data: "01/07/015" must not become year 15.
        assert pd.isna(parse_date("01/07/015"))

    def test_year_boundaries(self):
        assert parse_date("15/06/2020") == pd.Timestamp("2020-06-15")
        assert pd.isna(parse_date("01/01/1800"))
        assert pd.isna(parse_date("01/01/2200"))


class TestParseAmount:
    def test_simple_number(self):
        assert parse_amount("1000000") == 1000000.0

    def test_indian_format(self):
        assert parse_amount("20,00,00,000") == 200000000.0

    def test_with_dollar_sign(self):
        assert parse_amount("$500000") == 500000.0

    def test_undisclosed(self):
        assert pd.isna(parse_amount("undisclosed"))

    def test_unknown(self):
        assert pd.isna(parse_amount("unknown"))

    def test_empty(self):
        assert pd.isna(parse_amount(""))

    def test_negative(self):
        assert pd.isna(parse_amount("-100"))

    def test_nan(self):
        assert pd.isna(parse_amount(None))


class TestNormalizeColumnNames:
    def test_basic(self):
        df = pd.DataFrame({"Sr No": [1], "Date dd/mm/yyyy": ["01/01/2020"]})
        result = normalize_column_names(df)
        assert "sr_no" in result.columns
        assert "date_dd_mm_yyyy" in result.columns
