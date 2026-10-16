import re
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.database import create_database, get_db_summary, run_query
from src.preprocessing import preprocess

SQL_FILE = Path(__file__).resolve().parent.parent / "sql" / "analysis_queries.sql"


@pytest.fixture
def sample_df():
    raw = pd.DataFrame({
        "Sr No": [1, 2, 3],
        "Date dd/mm/yyyy": ["09/01/2020", "15/03/2019", "20/06/2018"],
        "Startup Name": ["CoA", "CoB", "CoC"],
        "Industry Vertical": ["E-commerce", "FinTech", "E-commerce"],
        "SubVertical": ["Marketplace", "Payments", "Retail"],
        "City  Location": ["Bangalore", "Mumbai", "Bangalore"],
        "Investors Name": ["Inv1", "Inv2", "Inv3"],
        "InvestmentnType": ["Seed Round", "Series A", "Series B"],
        "Amount in USD": ["10,00,000", "50,00,000", "1,00,00,000"],
        "Remarks": ["", "", ""],
    })
    return preprocess(raw)


@pytest.fixture
def db_path(tmp_path, sample_df):
    path = tmp_path / "test.db"
    create_database(sample_df, path)
    return path


class TestCreateDatabase:
    def test_creates_db(self, db_path):
        assert db_path.exists()

    def test_table_exists(self, db_path):
        result = run_query(db_path, "SELECT COUNT(*) as cnt FROM funding_events")
        assert result.iloc[0]["cnt"] == 3

    def test_views_exist(self, db_path):
        for view in ["v_yearly_summary", "v_sector_summary", "v_city_summary", "v_investment_type_summary"]:
            result = run_query(db_path, f"SELECT * FROM {view}")
            assert len(result) > 0


class TestRunQuery:
    def test_yearly_summary(self, db_path):
        result = run_query(db_path, "SELECT * FROM v_yearly_summary ORDER BY year")
        assert len(result) == 3
        assert result.iloc[0]["year"] == 2018

    def test_sector_summary(self, db_path):
        result = run_query(db_path, "SELECT * FROM v_sector_summary ORDER BY total_funding DESC")
        assert result.iloc[0]["industry_clean"] == "E-Commerce"

    def test_city_summary(self, db_path):
        result = run_query(db_path, "SELECT * FROM v_city_summary ORDER BY event_count DESC")
        assert result.iloc[0]["city_clean"] == "Bengaluru"


class TestGetDbSummary:
    def test_summary(self, db_path):
        summary = get_db_summary(db_path)
        assert summary["total_rows"] == 3
        assert summary["valid_amounts"] == 3
        assert summary["valid_dates"] == 3


def split_queries(text):
    """Split the analysis file into statements, dropping full-line comments."""
    body = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("--")
    )
    return [s.strip() for s in body.split(";") if s.strip()]


class TestAnalysisQueries:
    """Guards sql/analysis_queries.sql.

    Query 13 once referenced a column that does not exist (`amount_count`), so
    it raised "no such column" while the file still read as finished work. The
    file has no other test coverage, which is why it survived.
    """

    def test_file_exists(self):
        assert SQL_FILE.exists(), "sql/analysis_queries.sql is missing"

    def test_queries_are_found(self):
        queries = split_queries(SQL_FILE.read_text(encoding="utf-8"))
        assert len(queries) >= 13

    def test_every_query_runs(self, db_path):
        queries = split_queries(SQL_FILE.read_text(encoding="utf-8"))
        for i, query in enumerate(queries, 1):
            try:
                run_query(db_path, query)
            except Exception as exc:
                pytest.fail(f"query {i} did not run: {exc}\n\n{query}")

    def test_every_column_referenced_exists(self, db_path):
        """Catch a typo'd column even where the query would still parse."""
        con = sqlite3.connect(db_path)
        known = {r[1] for r in con.execute("PRAGMA table_info(funding_events)")}
        con.close()

        queries = split_queries(SQL_FILE.read_text(encoding="utf-8"))
        for i, query in enumerate(queries, 1):
            for ident in re.findall(r"\b([a-z_][a-z0-9_]*)\b", query.lower()):
                if ident in {"select", "from", "where", "and", "or", "not", "null",
                             "group", "by", "order", "limit", "as", "on", "join",
                             "desc", "asc", "in", "is", "between", "like", "case",
                             "when", "then", "else", "end", "sum", "count", "min",
                             "max", "avg", "round", "cast", "distinct", "having",
                             "coalesce", "strftime", "total", "over", "partition",
                             "all", "asc"}:
                    continue
                if ident in known:
                    continue
                # Only flag identifiers used in a column position.
                if re.search(rf"(?:count|sum|min|max|avg)\s*\(\s*{ident}\s*\)",
                             query.lower()):
                    pytest.fail(
                        f"query {i} aggregates unknown column {ident!r}\n\n{query}"
                    )
