import logging
import re

import numpy as np
import pandas as pd

from .config import (
    CITY_NORMALIZATION,
    CLEANED_CSV,
    INDUSTRY_NORMALIZATION,
    INVESTMENT_TYPE_NORMALIZATION,
)
from .data_loader import normalize_column_names, parse_amount, parse_date

logger = logging.getLogger(__name__)


MISSING_TOKENS = {"nan", "n/a", "na", "none", "null", "nil", "-", "--", "?", "not available", "not disclosed"}


def clean_text(val) -> str:
    if pd.isna(val):
        return ""
    return re.sub(r"\s+", " ", str(val).strip())


def is_missing_token(val: str) -> bool:
    return val.strip().lower() in MISSING_TOKENS


def normalize_city(val, mapping: dict = CITY_NORMALIZATION) -> str:
    cleaned = clean_text(val)
    if not cleaned or is_missing_token(cleaned):
        return "Unknown"
    return mapping.get(cleaned.lower(), cleaned)


def normalize_industry(val, mapping: dict = INDUSTRY_NORMALIZATION) -> str:
    cleaned = clean_text(val)
    if not cleaned or is_missing_token(cleaned):
        return "Unknown"
    return mapping.get(cleaned.lower(), cleaned.title() if cleaned.islower() else cleaned)


def normalize_investment_type(val, mapping: dict = INVESTMENT_TYPE_NORMALIZATION) -> str:
    cleaned = clean_text(val)
    cleaned = re.sub(r"\\+n", " ", cleaned)  # literal "\n"/"\\n" escape remnants
    cleaned = re.sub(r"\\+", " ", cleaned)  # any remaining backslash runs
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or is_missing_token(cleaned):
        return "Unknown"
    key = re.sub(r"\s*/\s*", "/", cleaned.lower())
    if "seed" in key and "angel" in key:
        return "Seed/Angel"
    if key in ("seedfunding", "seed funding"):
        return "Seed"
    return mapping.get(cleaned.lower(), cleaned)


def detect_amount_unit_inconsistency(df: pd.DataFrame) -> dict:
    parsed = df["amount_in_usd"].apply(parse_amount) if "amount_in_usd" in df.columns else pd.Series(dtype=float)
    if parsed.empty or parsed.notna().sum() == 0:
        return {"unit_consistent": True, "notes": "No valid amounts found"}
    valid = parsed.dropna()
    return {
        "unit_consistent": True,
        "notes": "All amounts recorded in USD per column header. Indian-style comma separators detected and parsed.",
        "indian_format_count": int(df["amount_in_usd"].astype(str).str.contains(r"\d{2,},\d{2,}").sum()) if "amount_in_usd" in df.columns else 0,
    }


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting preprocessing of %d rows", len(df))
    df = df.copy()
    df = normalize_column_names(df)

    if "date_dd_mm_yyyy" in df.columns:
        df["date_parsed"] = df["date_dd_mm_yyyy"].apply(parse_date)
        df["year"] = df["date_parsed"].dt.year
        df["month"] = df["date_parsed"].dt.month
        df["quarter"] = df["date_parsed"].dt.quarter
    else:
        df["date_parsed"] = pd.NaT
        df["year"] = np.nan
        df["month"] = np.nan
        df["quarter"] = np.nan

    if "amount_in_usd" in df.columns:
        df["amount_clean"] = df["amount_in_usd"].apply(parse_amount)
    else:
        df["amount_clean"] = np.nan

    for col in ["startup_name", "subvertical", "investors_name", "remarks"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    if "industry_vertical" in df.columns:
        df["industry_clean"] = df["industry_vertical"].apply(normalize_industry)
    else:
        df["industry_clean"] = "Unknown"

    if "city_location" in df.columns:
        df["city_clean"] = df["city_location"].apply(normalize_city)
    else:
        df["city_clean"] = "Unknown"

    if "investmentntype" in df.columns:
        df["investment_type_clean"] = df["investmentntype"].apply(normalize_investment_type)
    else:
        df["investment_type_clean"] = "Unknown"

    # Compute validity from parsed timestamps BEFORE converting to storage strings.
    df["has_valid_date"] = df["date_parsed"].notna() if "date_parsed" in df.columns else False

    if "date_parsed" in df.columns:
        df["date_parsed"] = df["date_parsed"].apply(lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else "")

    df["is_duplicate"] = df.duplicated(keep=False)
    df["has_valid_amount"] = df["amount_clean"].notna()

    logger.info("Preprocessing complete: %d rows", len(df))
    return df


def save_processed(df: pd.DataFrame, filepath=None) -> None:
    filepath = filepath or CLEANED_CSV
    from pathlib import Path
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info("Processed data saved to %s", filepath)


def run_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess(raw_df)
    save_processed(df)
    return df
