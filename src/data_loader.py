import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import QUALITY_REPORT, RAW_CSV

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "Sr No",
    "Date dd/mm/yyyy",
    "Startup Name",
    "Industry Vertical",
    "SubVertical",
    "City  Location",
    "Investors Name",
    "InvestmentnType",
    "Amount in USD",
    "Remarks",
]

SNA_VALUES = {"", "nan", "NaN", "null", "NULL", "N/A", "n/a", "NA", "None", "none", "undefined", "unknown", "UNKNOWN", "Undisclosed", "undisclosed", "Not Available", "not available", "TBD", "tbd", "-", "--"}


def load_raw_csv(filepath: Path | str = RAW_CSV) -> pd.DataFrame:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    df = pd.read_csv(filepath, encoding="utf-8", dtype=str, keep_default_na=False, na_values=[])
    return df


def validate_schema(df: pd.DataFrame) -> list[str]:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return []


MIN_VALID_YEAR = 1990
MAX_VALID_YEAR = 2030


def _is_in_valid_range(ts) -> bool:
    return pd.notna(ts) and MIN_VALID_YEAR <= ts.year <= MAX_VALID_YEAR


def parse_date(val):
    if pd.isna(val):
        return pd.NaT
    val = str(val).strip()
    if val in SNA_VALUES:
        return pd.NaT
    try:
        parsed = pd.to_datetime(val, format="%d/%m/%Y", errors="raise")
    except (ValueError, TypeError):
        try:
            parsed = pd.to_datetime(val, dayfirst=True, errors="raise")
        except (ValueError, TypeError):
            return pd.NaT
    # Reject out-of-range years (e.g. typos like "01/07/015" -> year 15).
    if not _is_in_valid_range(parsed):
        return pd.NaT
    return parsed


def parse_amount(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in SNA_VALUES:
        return np.nan
    s = s.replace(",", "").replace("$", "").replace("USD", "").strip()
    if s in SNA_VALUES:
        return np.nan
    try:
        result = float(s)
        if result < 0:
            return np.nan
        return result
    except (ValueError, TypeError):
        return np.nan


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        clean = (
            col.strip()
            .lower()
            .replace(" ", "_")
            .replace("__", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "")
        )
        rename_map[col] = clean
    return df.rename(columns=rename_map)


def generate_quality_report(df: pd.DataFrame) -> dict:
    report = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "duplicated_rows": int(df.duplicated().sum()),
        "missing_values": {},
        "data_types": {},
    }
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = round(missing_count / len(df) * 100, 2) if len(df) > 0 else 0
        report["missing_values"][col] = {"count": missing_count, "percentage": missing_pct}
        report["data_types"][col] = str(df[col].dtype)

    date_col = "date_dd_mm_yyyy" if "date_dd_mm_yyyy" in df.columns else None
    if date_col:
        parsed = df[date_col].apply(parse_date)
        report["date_range"] = {
            "min": str(parsed.min()) if not parsed.isna().all() else None,
            "max": str(parsed.max()) if not parsed.isna().all() else None,
            "invalid_dates": int(parsed.isna().sum()),
        }

    amount_col = "amount_in_usd" if "amount_in_usd" in df.columns else None
    if amount_col:
        parsed_amounts = df[amount_col].apply(parse_amount)
        report["amount_stats"] = {
            "valid_amounts": int(parsed_amounts.notna().sum()),
            "missing_amounts": int(parsed_amounts.isna().sum()),
            "min": float(parsed_amounts.min()) if parsed_amounts.notna().any() else None,
            "max": float(parsed_amounts.max()) if parsed_amounts.notna().any() else None,
            "mean": float(parsed_amounts.mean()) if parsed_amounts.notna().any() else None,
            "median": float(parsed_amounts.median()) if parsed_amounts.notna().any() else None,
        }
    return report


def save_quality_report(report: dict, filepath: Path | str = QUALITY_REPORT) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Quality report saved to %s", filepath)


def run_data_audit(filepath: Path | str = RAW_CSV) -> tuple[pd.DataFrame, dict]:
    logger.info("Starting data audit for %s", filepath)
    df = load_raw_csv(filepath)
    validate_schema(df)
    df = normalize_column_names(df)
    report = generate_quality_report(df)
    save_quality_report(report)
    logger.info("Data audit complete. %d rows, %d columns.", len(df), len(df.columns))
    return df, report
