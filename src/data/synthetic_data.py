"""
Synthetic EC transaction generator for local testing when URA API
credentials are not available.

Usage:
    python -m src.data.synthetic_data --rows 5000
"""

import argparse
import logging
import math
import os
import random
from datetime import date, timedelta

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_DSN = os.environ.get("DATABASE_URL", "postgresql://ec_user:ec_pass@localhost:5432/ec_db")

random.seed(42)

# Realistic EC projects (name, district, market_segment, commencement_year, total_units)
_PROJECTS = [
    ("The Criterion",            27, "OCR", 2016, 505),
    ("Parc Life",                27, "OCR", 2016, 628),
    ("The Terrace",              19, "OCR", 2015, 1040),
    ("The Vales",                19, "OCR", 2016, 517),
    ("Waterwoods",               19, "OCR", 2014, 583),
    ("Riverbank @ Fernvale",     28, "OCR", 2015, 555),
    ("Skypark Residences",       27, "OCR", 2015, 506),
    ("The Topiary",              28, "OCR", 2014, 700),
    ("Sol Acres",                23, "OCR", 2016, 1327),
    ("Wandervale",               23, "OCR", 2017, 534),
    ("Northwave",                25, "OCR", 2017, 358),
    ("Hundred Palms Residences", 19, "OCR", 2018, 531),
    ("Parc Canberra",            27, "OCR", 2020, 496),
    ("Piermont Grand",           19, "OCR", 2019, 820),
    ("Ola",                      19, "OCR", 2021, 548),
    ("Provence Residence",       27, "OCR", 2021, 413),
    ("North Gaia",               27, "OCR", 2022, 616),
    ("Tenet",                    18, "OCR", 2023, 618),
    ("Lumina Grand",             23, "OCR", 2024, 533),
    ("Novo Place",               23, "OCR", 2024, 504),
]

_FLOOR_RANGES = ["01-05","06-10","11-15","16-20","21-25","26-30"]
_AREAS_SQM = [72, 85, 99, 113, 125, 140]  # typical EC unit sizes


def _base_psf(commencement_year: int, district: int) -> float:
    """Approximate launch PSF based on year and district."""
    base = 700 + (commencement_year - 2013) * 35
    if district in (19, 28):
        base += 30
    return base + random.gauss(0, 40)


def _appreciation(years: float, district: int) -> float:
    """Annual appreciation rate (%) with some randomness."""
    rate = 0.045 + random.gauss(0, 0.008)
    return rate


def _generate_price(launch_psf: float, years: float, floor_mid: float, area_sqm: float) -> int:
    rate = _appreciation(years, 0)
    growth = (1 + rate) ** years
    floor_premium = 1 + 0.005 * max(floor_mid - 3, 0)
    psf = launch_psf * growth * floor_premium + random.gauss(0, 25)
    price_per_unit = psf * area_sqm * 10.7639
    return max(int(price_per_unit), 100_000)


def generate(n_rows: int = 5000) -> int:
    conn = psycopg2.connect(_DSN)
    conn.autocommit = False
    cur = conn.cursor()

    for proj_name, district, segment, comm_year, total_units in _PROJECTS:
        # Upsert project
        cur.execute(
            """
            INSERT INTO ec_projects (
                project_name, district, market_segment,
                tenure_years, lease_commencement_year, total_units,
                x_coord, y_coord
            ) VALUES (%s,%s,%s,99,%s,%s,%s,%s)
            ON CONFLICT (project_name) DO NOTHING
            """,
            (proj_name, district, segment, comm_year, total_units,
             24000 + random.uniform(-5000, 5000),
             30000 + random.uniform(-5000, 5000)),
        )

    conn.commit()

    rows_per_project = n_rows // len(_PROJECTS)
    inserted = 0

    for proj_name, district, segment, comm_year, _ in _PROJECTS:
        for _ in range(rows_per_project):
            years = random.uniform(0.5, 12)
            contract_date = date(comm_year, 1, 1) + timedelta(days=int(years * 365.25))
            if contract_date > date.today():
                contract_date = date.today()

            floor_range = random.choice(_FLOOR_RANGES)
            fl_low, fl_high = map(int, floor_range.split("-"))
            floor_mid = (fl_low + fl_high) / 2

            area_sqm = random.choice(_AREAS_SQM) + random.gauss(0, 3)
            launch_psf = _base_psf(comm_year, district)
            actual_years = (contract_date - date(comm_year, 1, 1)).days / 365.25
            price = _generate_price(launch_psf, actual_years, floor_mid, area_sqm)

            type_of_sale = 3 if actual_years > 2 else 1

            cur.execute(
                """
                INSERT INTO ec_transactions (
                    project_name, street, district, market_segment,
                    floor_range, area_sqm, type_of_sale, contract_date,
                    price, property_type, tenure, no_of_units
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1)
                """,
                (
                    proj_name,
                    f"Sample Street {district}",
                    district,
                    segment,
                    floor_range,
                    round(area_sqm, 2),
                    type_of_sale,
                    contract_date,
                    price,
                    "Executive Condominium",
                    f"99 years leasehold from {comm_year}",
                ),
            )
            inserted += 1

    conn.commit()
    cur.close()
    conn.close()
    log.info("Synthetic data: %d rows inserted.", inserted)
    return inserted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    args = parser.parse_args()
    generate(args.rows)
