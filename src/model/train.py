"""
Train a LightGBM model for EC price prediction at a given time horizon.

Two models are trained:
  - 5-year  (MOP)          : transactions at years_since_commencement ∈ [3.5, 6.5]
  - 10-year (privatisation) : transactions at years_since_commencement ∈ [8.5, 11.5]

Model selection rationale
--------------------------
LightGBM is chosen because:
1. Gradient-boosted trees are consistently best-in-class for tabular real-estate data.
2. It natively handles missing values (MRT distance often absent for older projects).
3. Built-in feature importance enables regulatory explainability.
4. Inference latency is <1 ms — suitable for a REST API.
5. Smaller training sets (thousands of EC transactions) favour trees over deep learning.

A 5-fold CV is used to report generalisation metrics before the final model
is retrained on all data and registered in the DB.
"""

import argparse
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_DSN = os.environ.get("DATABASE_URL", "postgresql://ec_user:ec_pass@localhost:5432/ec_db")
_MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
_MODEL_DIR.mkdir(exist_ok=True)

_WINDOW = 1.5  # ±1.5 years around target horizon

FEATURES = [
    "area_sqft",
    "floor_level_mid",
    "district",
    "market_segment_enc",
    "lease_commencement_year",
    "type_of_sale",
    "contract_year",
    "contract_quarter",
    "nearest_mrt_dist_m",
    "cbd_dist_m",
    "total_units_in_project",
    "years_since_commencement",
]

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 1000,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


def _load_data(horizon: int) -> pd.DataFrame:
    lo, hi = horizon - _WINDOW, horizon + _WINDOW
    sql = f"""
        SELECT price_psf, area_sqft, floor_level_mid, district,
               market_segment, lease_commencement_year, type_of_sale,
               contract_year, contract_quarter, nearest_mrt_dist_m,
               cbd_dist_m, total_units_in_project, years_since_commencement
        FROM   ec_features
        WHERE  years_since_commencement BETWEEN {lo} AND {hi}
          AND  price_psf IS NOT NULL AND price_psf > 100
    """
    conn = psycopg2.connect(_DSN)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def train(horizon: int) -> dict:
    log.info("=" * 60)
    log.info("Training %d-year EC price model", horizon)
    df = _load_data(horizon)

    if len(df) < 50:
        raise ValueError(
            f"Only {len(df)} rows for {horizon}-yr window. "
            "Run data ingestion first (POST /admin/ingest)."
        )
    log.info("Training rows: %d", len(df))

    le = LabelEncoder()
    df["market_segment_enc"] = le.fit_transform(df["market_segment"].fillna("OCR"))

    X = df[FEATURES].copy()
    y = df["price_psf"].values
    medians = X.median().to_dict()
    X = X.fillna(medians)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=[(X.iloc[val_idx], y[val_idx])],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        oof[val_idx] = m.predict(X.iloc[val_idx])
        fold_rmse = mean_squared_error(y[val_idx], oof[val_idx], squared=False)
        log.info("  Fold %d RMSE: %.2f", fold, fold_rmse)

    rmse = float(mean_squared_error(y, oof, squared=False))
    mape = _mape(y, oof)
    r2   = float(r2_score(y, oof))
    log.info("CV results — RMSE: %.2f | MAPE: %.2f%% | R²: %.4f", rmse, mape, r2)

    # Final model on all data
    final = lgb.LGBMRegressor(**LGBM_PARAMS)
    final.fit(X, y, callbacks=[lgb.log_evaluation(period=-1)])

    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name    = f"ec_lgbm_{horizon}yr_v{version}"
    path    = _MODEL_DIR / f"{name}.pkl"

    with open(path, "wb") as f:
        pickle.dump({
            "model": final,
            "label_encoder": le,
            "feature_columns": FEATURES,
            "medians": medians,
            "horizon": horizon,
        }, f)
    log.info("Model saved → %s", path)

    # Register and mark as deployed (deactivating old version)
    _register(name, horizon, version, str(path), rmse, mape, r2, len(df))
    return {"model": name, "rmse": rmse, "mape": mape, "r2": r2, "rows": len(df)}


def _register(name, horizon, version, path, rmse, mape, r2, rows):
    try:
        conn = psycopg2.connect(_DSN)
        cur  = conn.cursor()
        # Deactivate previous versions for this horizon
        cur.execute(
            "UPDATE model_registry SET deployed = FALSE WHERE target_horizon = %s",
            (horizon,),
        )
        cur.execute(
            """
            INSERT INTO model_registry
              (model_name, target_horizon, version, artifact_path,
               rmse, mape, r2, train_rows, hyperparameters, feature_list, deployed)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE)
            """,
            (name, horizon, version, path,
             rmse, mape, r2, rows,
             json.dumps(LGBM_PARAMS), json.dumps(FEATURES)),
        )
        conn.commit(); cur.close(); conn.close()
        log.info("Model registered and deployed.")
    except Exception as e:
        log.warning("Could not register model: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, choices=[5, 10], required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.horizon), indent=2))
