"""
EC Price Predictor — FastAPI application entry point.
"""

import logging
import os
import sys
import time

import psycopg2
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Allow src package imports regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from api.config import get_settings
from api.routes import health, predictions, admin

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Predicts Executive Condominium resale prices at **5-year MOP** and "
        "**10-year privatisation** milestones using a LightGBM model trained on "
        "historical URA private residential property transactions."
    ),
    contact={"email": "veennavena181@gmail.com"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request timing middleware ----------
@app.middleware("http")
async def add_timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Response-Time-ms"] = f"{ms:.1f}"
    _log_prediction(request, ms)
    return response


def _log_prediction(request: Request, ms: float):
    if request.url.path.startswith("/predict") and request.method == "POST":
        try:
            dsn = os.environ.get("DATABASE_URL", "")
            conn = psycopg2.connect(dsn)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO prediction_log (input_payload, latency_ms, client_ip) VALUES (%s, %s, %s)",
                ('{}', round(ms, 2), str(request.client.host if request.client else "")),
            )
            conn.commit(); cur.close(); conn.close()
        except Exception:
            pass


# ---------- Routes ----------
app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(admin.router)

# Serve the frontend at /ui
_frontend = os.path.join(os.path.dirname(__file__), "../frontend")
if os.path.isdir(_frontend):
    app.mount("/ui", StaticFiles(directory=_frontend, html=True), name="frontend")


# ---------- Startup ----------
@app.on_event("startup")
def startup():
    log.info("EC Price Predictor API starting …")
    _wait_for_db()
    _auto_train_if_needed()


def _wait_for_db(retries: int = 30):
    dsn = os.environ.get("DATABASE_URL", "")
    for i in range(retries):
        try:
            conn = psycopg2.connect(dsn)
            conn.close()
            log.info("Database ready.")
            return
        except Exception:
            log.warning("Waiting for DB … (%d/%d)", i + 1, retries)
            time.sleep(5)
    raise RuntimeError("Database not reachable after startup.")


def _auto_train_if_needed():
    """If no deployed model exists, run ingestion + training automatically."""
    try:
        dsn = os.environ.get("DATABASE_URL", "")
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM model_registry WHERE deployed = TRUE")
        count = cur.fetchone()[0]
        cur.close(); conn.close()

        if count == 0:
            log.info("No deployed models found — running auto-setup (ingest + train).")
            import threading
            threading.Thread(target=_run_setup, daemon=True).start()
        else:
            log.info("Found %d deployed model(s). Ready.", count)
    except Exception as e:
        log.warning("Could not check model registry: %s", e)


def _run_setup():
    try:
        from src.data.feature_engineering import build_features
        from src.model.train import train

        # Try real URA data first; fall back to synthetic if unreachable
        try:
            from src.data.ingestion import ingest
            log.info("Auto-setup: ingesting real URA data …")
            ingest()
        except Exception as ura_err:
            log.warning("URA API unreachable (%s). Using synthetic data for local dev.", ura_err)
            from src.data.synthetic_data import generate
            generate(5000)

        build_features()
        log.info("Auto-setup: training models …")
        train(5)
        train(10)
        log.info("Auto-setup complete. API ready for predictions.")
    except Exception:
        log.exception("Auto-setup failed. Check logs above.")
