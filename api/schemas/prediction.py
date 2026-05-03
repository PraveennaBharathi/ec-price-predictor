from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    area_sqft: float = Field(..., gt=0, le=10000, description="Unit area in square feet")
    floor_level: float = Field(..., ge=1, le=50, description="Floor level (e.g. 8 for floor 06-10)")
    district: int = Field(..., ge=1, le=28, description="Singapore district number (1-28)")
    market_segment: str = Field(..., pattern="^(CCR|RCR|OCR)$", description="CCR, RCR, or OCR")
    lease_commencement_year: int = Field(..., ge=1990, le=2030, description="Year lease commenced")
    nearest_mrt_dist_m: Optional[float] = Field(None, ge=0, description="Distance to nearest MRT in metres")
    cbd_dist_m: Optional[float] = Field(None, ge=0, description="Distance to CBD (Raffles Place) in metres")
    total_units: Optional[int] = Field(None, ge=1, description="Total units in development")

    model_config = {
        "json_schema_extra": {
            "example": {
                "area_sqft": 1076,
                "floor_level": 8,
                "district": 19,
                "market_segment": "OCR",
                "lease_commencement_year": 2019,
                "nearest_mrt_dist_m": 450,
                "cbd_dist_m": 16000,
                "total_units": 820
            }
        }
    }


class HorizonResult(BaseModel):
    psf: float = Field(description="Predicted price per square foot (SGD)")
    total_price_est: int = Field(description="Estimated total price (SGD)")
    model_version: str
    model_rmse: float = Field(description="Cross-validated RMSE on training data")
    model_mape: float = Field(description="Cross-validated MAPE % on training data")
    model_r2: float = Field(description="Cross-validated R² on training data")


class PredictionResponse(BaseModel):
    input: PredictionRequest
    mop_5yr: HorizonResult = Field(description="Predicted price at 5-year MOP")
    privatisation_10yr: HorizonResult = Field(description="Predicted price at 10-year privatisation")
    note: str = "Predictions are indicative estimates based on historical EC transactions."


class ModelInfoResponse(BaseModel):
    horizon: int
    model_name: str
    version: str
    artifact_path: str
    rmse: float
    mape: float
    r2: float
    train_rows: int
    trained_at: str
    deployed: bool
