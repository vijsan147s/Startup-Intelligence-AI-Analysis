import logging
import sqlite3
from pathlib import Path

import pandas as pd

from .config import DB_PATH

logger = logging.getLogger(__name__)


def create_database(df: pd.DataFrame, db_path=None) -> Path:
    db_path = Path(db_path) if db_path else DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        df.to_sql("funding_events", conn, index=False, if_exists="replace")
        _create_views(conn)
        conn.commit()
        logger.info("Database created at %s with %d rows", db_path, len(df))
    finally:
        conn.close()
    return db_path


def _create_views(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_yearly_summary AS
        SELECT
            year,
            COUNT(*) AS event_count,
            SUM(amount_clean) AS total_funding,
            AVG(amount_clean) AS avg_funding,
            COUNT(amount_clean) AS events_with_amount
        FROM funding_events
        WHERE year IS NOT NULL
        GROUP BY year
        ORDER BY year
    """)
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_sector_summary AS
        SELECT
            industry_clean,
            COUNT(*) AS event_count,
            SUM(amount_clean) AS total_funding,
            AVG(amount_clean) AS avg_funding,
            COUNT(amount_clean) AS events_with_amount
        FROM funding_events
        GROUP BY industry_clean
        ORDER BY total_funding DESC
    """)
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_city_summary AS
        SELECT
            city_clean,
            COUNT(*) AS event_count,
            SUM(amount_clean) AS total_funding,
            AVG(amount_clean) AS avg_funding,
            COUNT(amount_clean) AS events_with_amount
        FROM funding_events
        GROUP BY city_clean
        ORDER BY total_funding DESC
    """)
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_investment_type_summary AS
        SELECT
            investment_type_clean,
            COUNT(*) AS event_count,
            SUM(amount_clean) AS total_funding,
            AVG(amount_clean) AS avg_funding,
            COUNT(amount_clean) AS events_with_amount
        FROM funding_events
        GROUP BY investment_type_clean
        ORDER BY event_count DESC
    """)


def run_query(db_path, query: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def get_db_summary(db_path=None) -> dict:
    db_path = Path(db_path) if db_path else DB_PATH
    conn = sqlite3.connect(db_path)
    try:
        total_rows = conn.execute("SELECT COUNT(*) FROM funding_events").fetchone()[0]
        valid_amounts = conn.execute("SELECT COUNT(amount_clean) FROM funding_events WHERE amount_clean IS NOT NULL").fetchone()[0]
        valid_dates = conn.execute("SELECT COUNT(date_parsed) FROM funding_events WHERE date_parsed IS NOT NULL AND date_parsed != ''").fetchone()[0]
        total_funding = conn.execute("SELECT SUM(amount_clean) FROM funding_events WHERE amount_clean IS NOT NULL").fetchone()[0]
        date_range = conn.execute("SELECT MIN(date_parsed), MAX(date_parsed) FROM funding_events WHERE date_parsed IS NOT NULL AND date_parsed != ''").fetchone()
        return {
            "total_rows": total_rows,
            "valid_amounts": valid_amounts,
            "valid_dates": valid_dates,
            "total_funding": total_funding,
            "date_min": date_range[0],
            "date_max": date_range[1],
        }
    finally:
        conn.close()
