"""Bridge between API layer and ML inference layer."""

import logging
import sys
import os

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from src.model.predict import predict, invalidate_cache
from api.schemas.prediction import PredictionRequest, PredictionResponse

log = logging.getLogger(__name__)


def run_prediction(req: PredictionRequest) -> PredictionResponse:
    raw = predict(
        area_sqft=req.area_sqft,
        floor_level=req.floor_level,
        district=req.district,
        market_segment=req.market_segment,
        lease_commencement_year=req.lease_commencement_year,
        nearest_mrt_dist_m=req.nearest_mrt_dist_m,
        cbd_dist_m=req.cbd_dist_m,
        total_units=req.total_units,
    )
    from api.schemas.prediction import HorizonResult
    return PredictionResponse(
        input=req,
        mop_5yr=HorizonResult(**raw["mop_5yr"]),
        privatisation_10yr=HorizonResult(**raw["privatisation_10yr"]),
    )


def clear_model_cache() -> None:
    invalidate_cache()
