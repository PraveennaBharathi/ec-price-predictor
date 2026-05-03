"""
Admin endpoints — protected by a simple bearer token.
In production replace with proper auth (OAuth2 / API key management).
"""

import logging
import os
import sys
import threading

from fastapi import APIRouter, Depends, HTTPException, Header
from typing import Annotated

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

log = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])

_ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "admin-secret")


def _require_admin(x_admin_token: Annotated[str, Header()] = ""):
    if x_admin_token != _ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")


@router.post("/ingest", dependencies=[Depends(_require_admin)], summary="Trigger URA data ingestion")
def trigger_ingest(background: bool = True):
    """
    Pull fresh transaction data from URA API → PostgreSQL.
    Set background=false to wait for completion (slow, up to several minutes).
    """
    from src.data.ingestion import ingest
    from src.data.feature_engineering import build_features

    def _run():
        try:
            n = ingest()
            build_features()
            log.info("Ingestion+features done: %d rows", n)
        except Exception:
            log.exception("Ingestion failed")

    if background:
        threading.Thread(target=_run, daemon=True).start()
        return {"status": "started", "message": "Ingestion running in background."}
    else:
        from src.data.ingestion import ingest
        from src.data.feature_engineering import build_features
        n = ingest()
        built = build_features()
        return {"status": "done", "ingested": n, "features": built}


@router.post("/train", dependencies=[Depends(_require_admin)], summary="Trigger model (re)training")
def trigger_train(horizon: int = 0, background: bool = True):
    """
    Train LightGBM models.
    horizon=0 trains both 5-yr and 10-yr models.
    horizon=5 or horizon=10 trains only that model.
    """
    from src.model.train import train
    from api.services.prediction_service import clear_model_cache

    horizons = [5, 10] if horizon == 0 else [horizon]
    if any(h not in (5, 10) for h in horizons):
        raise HTTPException(400, "horizon must be 0, 5, or 10")

    def _run():
        try:
            for h in horizons:
                result = train(h)
                log.info("Training done: %s", result)
            clear_model_cache()
        except Exception:
            log.exception("Training failed")

    if background:
        threading.Thread(target=_run, daemon=True).start()
        return {"status": "started", "horizons": horizons}
    else:
        results = {}
        for h in horizons:
            results[f"{h}yr"] = train(h)
        clear_model_cache()
        return {"status": "done", "results": results}


@router.get("/monitoring", dependencies=[Depends(_require_admin)], summary="Recent monitoring metrics")
def get_monitoring(limit: int = 50):
    import psycopg2, psycopg2.extras
    dsn = os.environ.get("DATABASE_URL", "")
    conn = psycopg2.connect(dsn)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT m.metric_name, m.metric_value, m.window_start, m.window_end,
               m.recorded_at, r.model_name, r.target_horizon
        FROM   model_monitoring m
        JOIN   model_registry   r ON r.id = m.model_id
        ORDER  BY m.recorded_at DESC
        LIMIT  %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close(); conn.close()
    return {"metrics": [dict(r) for r in rows]}
