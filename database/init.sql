-- ============================================================
-- EC Price Predictor – PostgreSQL Schema
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -------------------------------------------------------
-- 1. Raw URA transactions (one row per transacted unit)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS ec_transactions (
    id               UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_name     TEXT        NOT NULL,
    street           TEXT,
    district         SMALLINT,
    market_segment   VARCHAR(10),          -- CCR / RCR / OCR
    floor_range      VARCHAR(20),          -- e.g. "06-10"
    area_sqm         NUMERIC(10,2),
    type_of_sale     SMALLINT,             -- 1=New, 2=SubSale, 3=Resale
    contract_date    DATE        NOT NULL,
    price            BIGINT      NOT NULL, -- total transacted price (SGD)
    nett_price       BIGINT,               -- nett price if available
    property_type    VARCHAR(50),          -- "Executive Condominium" etc.
    tenure           TEXT,                 -- "99 years leasehold from 2015"
    no_of_units      SMALLINT    DEFAULT 1,
    x_coord          NUMERIC(12,4),
    y_coord          NUMERIC(12,4),
    ingested_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ec_tx_project   ON ec_transactions(project_name);
CREATE INDEX IF NOT EXISTS idx_ec_tx_date      ON ec_transactions(contract_date);
CREATE INDEX IF NOT EXISTS idx_ec_tx_district  ON ec_transactions(district);

-- -------------------------------------------------------
-- 2. EC project master (one row per development)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS ec_projects (
    id                      UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_name            TEXT    UNIQUE NOT NULL,
    street                  TEXT,
    district                SMALLINT,
    market_segment          VARCHAR(10),
    tenure_years            SMALLINT,
    lease_commencement_year SMALLINT,
    total_units             SMALLINT,
    x_coord                 NUMERIC(12,4),
    y_coord                 NUMERIC(12,4),
    nearest_mrt_name        TEXT,
    nearest_mrt_dist_m      NUMERIC(8,2),
    cbd_dist_m              NUMERIC(8,2),
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

-- -------------------------------------------------------
-- 3. Engineered feature set (materialised for ML training)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS ec_features (
    id                          UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id              UUID    REFERENCES ec_transactions(id) ON DELETE CASCADE,
    project_id                  UUID    REFERENCES ec_projects(id)     ON DELETE CASCADE,
    price_psf                   NUMERIC(10,2) NOT NULL,  -- SGD per sqft
    area_sqft                   NUMERIC(10,2),
    floor_level_mid             NUMERIC(6,2),            -- midpoint of floor range
    years_since_commencement    NUMERIC(5,2),            -- at transaction date
    district                    SMALLINT,
    market_segment              VARCHAR(10),
    lease_commencement_year     SMALLINT,
    type_of_sale                SMALLINT,
    contract_year               SMALLINT,
    contract_quarter            SMALLINT,
    nearest_mrt_dist_m          NUMERIC(8,2),
    cbd_dist_m                  NUMERIC(8,2),
    total_units_in_project      SMALLINT,
    created_at                  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feat_years ON ec_features(years_since_commencement);
CREATE INDEX IF NOT EXISTS idx_feat_proj  ON ec_features(project_id);

-- -------------------------------------------------------
-- 4. Trained model registry
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_registry (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name      VARCHAR(100) NOT NULL,   -- e.g. "ec_lgbm_5yr_v1"
    target_horizon  SMALLINT    NOT NULL,    -- 5 or 10 (years)
    version         VARCHAR(20) NOT NULL,
    artifact_path   TEXT        NOT NULL,    -- path / S3 URI to serialised model
    rmse            NUMERIC(10,4),
    mape            NUMERIC(8,4),
    r2              NUMERIC(8,4),
    train_rows      INTEGER,
    feature_list    JSONB,
    hyperparameters JSONB,
    trained_at      TIMESTAMPTZ DEFAULT NOW(),
    deployed        BOOLEAN     DEFAULT FALSE,
    UNIQUE(model_name, version)
);

-- -------------------------------------------------------
-- 5. Prediction log (audit trail + monitoring)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS prediction_log (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id        UUID        REFERENCES model_registry(id),
    input_payload   JSONB       NOT NULL,
    prediction_5yr  NUMERIC(10,2),
    prediction_10yr NUMERIC(10,2),
    latency_ms      NUMERIC(8,2),
    requested_at    TIMESTAMPTZ DEFAULT NOW(),
    client_ip       INET
);

-- -------------------------------------------------------
-- 6. Model monitoring metrics (time-series)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_monitoring (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id        UUID        REFERENCES model_registry(id),
    metric_name     VARCHAR(100) NOT NULL,  -- "psi", "mae", "feature_drift_floor_level"
    metric_value    NUMERIC(14,6),
    window_start    DATE,
    window_end      DATE,
    recorded_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mon_model   ON model_monitoring(model_id);
CREATE INDEX IF NOT EXISTS idx_mon_metric  ON model_monitoring(metric_name);
CREATE INDEX IF NOT EXISTS idx_mon_date    ON model_monitoring(recorded_at);

-- -------------------------------------------------------
-- 7. Seed: district → region lookup
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS district_region (
    district    SMALLINT PRIMARY KEY,
    region      VARCHAR(10) NOT NULL,  -- CCR / RCR / OCR
    area_name   TEXT
);

INSERT INTO district_region VALUES
 (1,'CCR','Boat Quay / Raffles Place / Marina'),
 (2,'CCR','Chinatown / Tanjong Pagar'),
 (3,'RCR','Queenstown / Tiong Bahru'),
 (4,'RCR','Telok Blangah / Harbourfront'),
 (5,'RCR','Pasir Panjang / Clementi'),
 (6,'CCR','City Hall / Clarke Quay'),
 (7,'CCR','Beach Road / Bugis / Rochor'),
 (8,'RCR','Farrer Park / Serangoon Rd'),
 (9,'CCR','Orchard / River Valley'),
 (10,'CCR','Bukit Timah / Holland Rd'),
 (11,'CCR','Newton / Novena'),
 (12,'RCR','Balestier / Toa Payoh / Serangoon'),
 (13,'RCR','Macpherson / Braddell'),
 (14,'RCR','Geylang / Eunos'),
 (15,'RCR','Katong / Joo Chiat / Amber'),
 (16,'OCR','Bedok / Upper East Coast'),
 (17,'OCR','Loyang / Changi'),
 (18,'OCR','Tampines / Pasir Ris'),
 (19,'OCR','Serangoon Gardens / Hougang / Punggol'),
 (20,'OCR','Bishan / Ang Mo Kio'),
 (21,'OCR','Upper Bukit Timah / Ulu Pandan'),
 (22,'OCR','Boon Lay / Jurong / Tuas'),
 (23,'OCR','Hillview / Bukit Panjang / Choa Chu Kang'),
 (24,'OCR','Lim Chu Kang / Tengah'),
 (25,'OCR','Admiralty / Woodlands'),
 (26,'OCR','Mandai / Upper Thomson'),
 (27,'OCR','Sembawang / Yishun'),
 (28,'OCR','Seletar / Yio Chu Kang')
ON CONFLICT DO NOTHING;
