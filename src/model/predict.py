"""
Inference wrapper — loads the latest deployed model from DB and runs predictions.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

_DSN = os.environ.get("DATABASE_URL", "postgresql://ec_user:ec_pass@localhost:5432/ec_db")
_MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))

# Cache loaded models in memory so we don't re-read pickle every request
_cache: dict[int, dict] = {}


def _load_model(horizon: Literal[5, 10]) -> dict:
    if horizon in _cache:
        return _cache[horizon]

    conn = psycopg2.connect(_DSN)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute(
        """
        SELECT artifact_path, feature_list, rmse, mape, r2, version
        FROM   model_registry
        WHERE  target_horizon = %s AND deployed = TRUE
        ORDER  BY trained_at DESC
        LIMIT  1
        """,
        (horizon,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row is None:
        raise FileNotFoundError(f"No deployed model found for horizon={horizon}yr. Run training first.")

    with open(row["artifact_path"], "rb") as f:
        payload = pickle.load(f)

    payload["db_meta"] = dict(row)
    _cache[horizon] = payload
    log.info("Loaded %d-yr model v%s (RMSE=%.2f)", horizon, row["version"], row["rmse"])
    return payload


def predict(
    *,
    area_sqft: float,
    floor_level: float,
    district: int,
    market_segment: str,
    lease_commencement_year: int,
    nearest_mrt_dist_m: float | None = None,
    cbd_dist_m: float | None = None,
    total_units: int | None = None,
    type_of_sale: int = 3,
) -> dict:
    """
    Return predictions for both 5-yr MOP and 10-yr privatisation.

    Returns a dict with:
        mop_psf, privatisation_psf, mop_price_est, privatisation_price_est
        plus model metadata.
    """
    results = {}
    for horizon in (5, 10):
        payload = _load_model(horizon)
        le = payload["label_encoder"]
        medians: dict = payload["medians"]
        features: list[str] = payload["feature_columns"]
        model = payload["model"]

        seg_enc = le.transform([market_segment])[0] if market_segment in le.classes_ else 0

        row = {
            "area_sqft": area_sqft,
            "floor_level_mid": floor_level,
            "district": district,
            "market_segment_enc": seg_enc,
            "lease_commencement_year": lease_commencement_year,
            "type_of_sale": type_of_sale,
            "contract_year": lease_commencement_year + horizon,
            "contract_quarter": 2,
            "nearest_mrt_dist_m": nearest_mrt_dist_m if nearest_mrt_dist_m is not None else medians.get("nearest_mrt_dist_m", 500),
            "cbd_dist_m": cbd_dist_m if cbd_dist_m is not None else medians.get("cbd_dist_m", 15000),
            "total_units_in_project": total_units if total_units is not None else medians.get("total_units_in_project", 500),
            "years_since_commencement": float(horizon),
        }

        X = pd.DataFrame([row])[features]
        for col in features:
            if X[col].isna().any():
                X[col] = medians.get(col, 0)

        psf = float(model.predict(X)[0])
        price_est = psf * area_sqft

        meta = payload["db_meta"]
        results[horizon] = {
            "psf": round(psf, 2),
            "total_price_est": round(price_est),
            "model_version": meta["version"],
            "model_rmse": round(float(meta["rmse"]), 2),
            "model_mape": round(float(meta["mape"]), 2),
            "model_r2": round(float(meta["r2"]), 4),
        }

    return {
        "mop_5yr": results[5],
        "privatisation_10yr": results[10],
    }


def invalidate_cache() -> None:
    """Call after retraining to force reload on next prediction."""
    _cache.clear()
