from fastapi import APIRouter
from api.database import check_connection

router = APIRouter(tags=["Health"])


@router.get("/health")
def health():
    db_ok = check_connection()
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "unreachable",
    }


@router.get("/")
def root():
    return {"message": "EC Price Predictor API", "docs": "/docs"}
