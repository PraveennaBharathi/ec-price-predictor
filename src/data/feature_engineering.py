"""
Feature engineering: transform raw ec_transactions + ec_projects
into the ec_features table used for model training.

Key transforms:
- price_psf       = (price / no_of_units) / area_sqft
- floor_level_mid = midpoint of e.g. "06-10"
- years_since_commencement = (contract_date - lease_commencement_year-01-01) / 365.25
- CBD distance    = Euclidean approx (SVY21 coords)
"""

import logging
import math
import os
import re

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_DSN = os.environ.get("DATABASE_URL", "postgresql://ec_user:ec_pass@localhost:5432/ec_db")

# Raffles Place in SVY21 approx coordinates
_CBD_X = 29_000.0
_CBD_Y = 29_500.0
_SQM_TO_SQFT = 10.7639


def _floor_midpoint(floor_range: str) -> float | None:
    """'06-10' → 8.0,  'B1-01' → None (basement/special)."""
    m = re.match(r"(\d+)-(\d+)", floor_range or "")
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2
    return None


def _euclidean_dist_m(x1, y1, x2, y2) -> float | None:
    if None in (x1, y1, x2, y2):
        return None
    return math.sqrt((float(x1) - x2) ** 2 + (float(y1) - y2) ** 2)


def _years_since_commencement(contract_date, commencement_year: int | None) -> float | None:
    if commencement_year is None or contract_date is None:
        return None
    commencement = f"{commencement_year}-01-01"
    try:
        from datetime import date
        comm = date(commencement_year, 1, 1)
        delta = (contract_date - comm).days
        return round(delta / 365.25, 3)
    except Exception:
        return None


_FETCH_SQL = """
SELECT
    t.id            AS transaction_id,
    p.id            AS project_id,
    t.price,
    t.no_of_units,
    t.area_sqm,
    t.floor_range,
    t.contract_date,
    t.district,
    t.market_segment,
    t.type_of_sale,
    p.lease_commencement_year,
    p.nearest_mrt_dist_m,
    p.cbd_dist_m,
    p.total_units   AS total_units_in_project,
    p.x_coord,
    p.y_coord
FROM ec_transactions  t
JOIN ec_projects      p ON p.project_name = t.project_name
WHERE t.price IS NOT NULL
  AND t.area_sqm IS NOT NULL
  AND t.area_sqm > 0
"""

_UPSERT_FEATURE = """
INSERT INTO ec_features (
    transaction_id, project_id,
    price_psf, area_sqft, floor_level_mid,
    years_since_commencement, district, market_segment,
    lease_commencement_year, type_of_sale,
    contract_year, contract_quarter,
    nearest_mrt_dist_m, cbd_dist_m,
    total_units_in_project
) VALUES (
    %(transaction_id)s, %(project_id)s,
    %(price_psf)s, %(area_sqft)s, %(floor_level_mid)s,
    %(years_since_commencement)s, %(district)s, %(market_segment)s,
    %(lease_commencement_year)s, %(type_of_sale)s,
    %(contract_year)s, %(contract_quarter)s,
    %(nearest_mrt_dist_m)s, %(cbd_dist_m)s,
    %(total_units_in_project)s
)
ON CONFLICT DO NOTHING
"""


def build_features() -> int:
    conn = psycopg2.connect(_DSN)
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute(_FETCH_SQL)
    rows = cur.fetchall()

    written = 0
    for row in rows:
        area_sqft = float(row["area_sqm"] or 0) * _SQM_TO_SQFT
        if area_sqft == 0:
            continue

        per_unit_price = float(row["price"] or 0) / max(int(row["no_of_units"] or 1), 1)
        price_psf = per_unit_price / area_sqft

        cbd_dist = float(row["cbd_dist_m"]) if row["cbd_dist_m"] else _euclidean_dist_m(
            row["x_coord"], row["y_coord"], _CBD_X, _CBD_Y
        )
        cd = row["contract_date"]
        ysc = _years_since_commencement(cd, row["lease_commencement_year"])

        feat = {
            "transaction_id": row["transaction_id"],
            "project_id": row["project_id"],
            "price_psf": round(price_psf, 2),
            "area_sqft": round(area_sqft, 2),
            "floor_level_mid": _floor_midpoint(row["floor_range"]),
            "years_since_commencement": ysc,
            "district": row["district"],
            "market_segment": row["market_segment"],
            "lease_commencement_year": row["lease_commencement_year"],
            "type_of_sale": row["type_of_sale"],
            "contract_year": cd.year if cd else None,
            "contract_quarter": ((cd.month - 1) // 3 + 1) if cd else None,
            "nearest_mrt_dist_m": row["nearest_mrt_dist_m"],
            "cbd_dist_m": cbd_dist,
            "total_units_in_project": row["total_units_in_project"],
        }

        write_cur = conn.cursor()
        write_cur.execute(_UPSERT_FEATURE, feat)
        write_cur.close()
        written += 1

    conn.commit()
    cur.close()
    conn.close()
    log.info("Feature engineering complete — %d rows written.", written)
    return written


if __name__ == "__main__":
    build_features()
