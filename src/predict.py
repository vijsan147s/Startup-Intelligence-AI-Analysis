import logging

import numpy as np
import pandas as pd

from .config import (
    MAX_PREDICTABLE_YEAR,
    MIN_PREDICTABLE_YEAR,
    MODELS_DIR,
    USD_TO_INR,
    USD_TO_INR_CRORE_DIVISOR,
)
from .features import FEATURE_COLUMNS
from .train import load_pipeline

logger = logging.getLogger(__name__)


class FundingPredictor:
    def __init__(self, pipeline_path=None):
        self.pipeline = load_pipeline(pipeline_path or MODELS_DIR / "funding_model_pipeline.joblib")
        self.feature_names = list(FEATURE_COLUMNS)

    def predict(self, industry: str, subvertical: str, city: str, investment_type: str,
                year: int, month: int, investors_name: str = "") -> dict:
        if year < MIN_PREDICTABLE_YEAR or year > MAX_PREDICTABLE_YEAR:
            raise ValueError(
                f"Year {year} is outside the supported range "
                f"({MIN_PREDICTABLE_YEAR}-{MAX_PREDICTABLE_YEAR})"
            )
        if month < 1 or month > 12:
            raise ValueError(f"Month {month} must be between 1 and 12")

        row = {
            "industry_clean": industry if industry else "Unknown",
            "subvertical": subvertical if subvertical else "Unknown",
            "city_clean": city if city else "Unknown",
            "investment_type_clean": investment_type if investment_type else "Unknown",
            "year": float(year),
            "month": float(month),
            "investors_name": investors_name if investors_name else "",
        }
        input_df = pd.DataFrame([row])

        prediction = self.pipeline.predict(input_df)[0]
        prediction = max(0, float(prediction))

        return {
            "estimated_amount_usd": round(prediction, 2),
            "estimated_amount_inr_crores": round(
                prediction * USD_TO_INR / USD_TO_INR_CRORE_DIVISOR, 2
            ),
            "inputs": {
                "industry": industry,
                "subvertical": subvertical,
                "city": city,
                "investment_type": investment_type,
                "year": year,
                "month": month,
                "investors": investors_name,
            },
            "disclaimer": (
                "This is a model-based estimate from historical data (2015-2020). "
                "It does not guarantee actual funding or predict investment success."
            ),
        }

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(df)
