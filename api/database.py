import os
import psycopg2
import psycopg2.extras
from contextlib import contextmanager

_DSN = os.environ.get("DATABASE_URL", "postgresql://ec_user:ec_pass@localhost:5432/ec_db")


@contextmanager
def get_conn():
    conn = psycopg2.connect(_DSN)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def check_connection() -> bool:
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            return True
    except Exception:
        return False
