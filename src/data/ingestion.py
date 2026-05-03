"""
Data ingestion pipeline: URA API → PostgreSQL.
Run directly:  python -m src.data.ingestion
"""

import logging
import os
from datetime import date

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from src.data.ura_client import URAClient, parse_contract_date

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://ec_user:ec_pass@localhost:5432/ec_db",
)

_INSERT_TX = """
    INSERT INTO ec_transactions (
        project_name, street, district, market_segment,
        floor_range, area_sqm, type_of_sale, contract_date,
        price, nett_price, property_type, tenure,
        no_of_units, x_coord, y_coord
    ) VALUES (
        %(project_name)s, %(street)s, %(district)s, %(market_segment)s,
        %(floor_range)s, %(area_sqm)s, %(type_of_sale)s, %(contract_date)s,
        %(price)s, %(nett_price)s, %(property_type)s, %(tenure)s,
        %(no_of_units)s, %(x_coord)s, %(y_coord)s
    )
    ON CONFLICT DO NOTHING
"""


def _parse_lease_year(tenure: str) -> int | None:
    """Extract commencement year from e.g. '99 years leasehold from 2018'."""
    try:
        return int(tenure.split("from")[-1].strip().split()[0])
    except Exception:
        return None


def _upsert_project(cur, row: dict, commencement_year: int | None) -> None:
    cur.execute(
        """
        INSERT INTO ec_projects (
            project_name, street, district, market_segment,
            tenure_years, lease_commencement_year, x_coord, y_coord
        ) VALUES (
            %(project_name)s, %(street)s, %(district)s, %(market_segment)s,
            99, %(commencement_year)s, %(x_coord)s, %(y_coord)s
        )
        ON CONFLICT (project_name) DO UPDATE
            SET district             = EXCLUDED.district,
                market_segment       = EXCLUDED.market_segment,
                lease_commencement_year = COALESCE(
                    ec_projects.lease_commencement_year,
                    EXCLUDED.lease_commencement_year
                ),
                updated_at = NOW()
        """,
        {**row, "commencement_year": commencement_year},
    )


def ingest(dry_run: bool = False) -> int:
    with URAClient() as client:
        raw_rows = client.fetch_ec_transactions()

    if dry_run:
        log.info("Dry run — %d rows would be inserted.", len(raw_rows))
        return len(raw_rows)

    conn = psycopg2.connect(_DSN)
    conn.autocommit = False
    cur = conn.cursor()

    inserted = 0
    for row in raw_rows:
        contract_date: date | None = parse_contract_date(row["contract_date"])
        if contract_date is None:
            log.warning("Skipping row with unparseable date: %s", row["contract_date"])
            continue

        commencement_year = _parse_lease_year(row.get("tenure", ""))
        _upsert_project(cur, row, commencement_year)

        cur.execute(
            _INSERT_TX,
            {**row, "contract_date": contract_date},
        )
        inserted += 1

    conn.commit()
    cur.close()
    conn.close()
    log.info("Ingestion complete — %d rows inserted / updated.", inserted)
    return inserted


if __name__ == "__main__":
    ingest()
