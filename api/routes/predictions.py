import logging
from fastapi import APIRouter, HTTPException
from api.schemas.prediction import PredictionRequest, PredictionResponse
from api.services.prediction_service import run_prediction

log = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Predictions"])


@router.post("", response_model=PredictionResponse, summary="Predict EC prices at MOP and privatisation")
def predict_price(req: PredictionRequest):
    """
    Predict the resale price per sqft (and estimated total price) of an
    Executive Condominium unit at two key milestones:

    - **5 years** after lease commencement (Minimum Occupancy Period)
    - **10 years** after lease commencement (Privatisation)
    """
    try:
        return run_prediction(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@router.get("/model/info", summary="Metadata for deployed models")
def model_info():
    import psycopg2, psycopg2.extras, os
    dsn = os.environ.get("DATABASE_URL", "postgresql://ec_user:ec_pass@localhost:5432/ec_db")
    conn = psycopg2.connect(dsn)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT model_name, target_horizon, version, rmse, mape, r2,
               train_rows, trained_at, deployed, feature_list
        FROM   model_registry
        WHERE  deployed = TRUE
        ORDER  BY target_horizon, trained_at DESC
    """)
    rows = cur.fetchall()
    cur.close(); conn.close()
    return {"deployed_models": [dict(r) for r in rows]}
