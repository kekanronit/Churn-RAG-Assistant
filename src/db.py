
import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "churn.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database():
    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                gender TEXT,
                senior_citizen INTEGER,
                partner TEXT,
                dependents TEXT,
                tenure_months INTEGER,
                contract TEXT,
                payment_method TEXT,
                paperless_billing TEXT,
                internet_service TEXT,
                monthly_charges REAL,
                total_charges REAL,
                churn_label TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS support_tickets (
                ticket_id INTEGER PRIMARY KEY,
                customer_id TEXT NOT NULL,
                category TEXT NOT NULL,
                created_at TEXT NOT NULL,
                resolved INTEGER NOT NULL,
                notes TEXT,
                FOREIGN KEY (customer_id)
                    REFERENCES customers(customer_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                churn_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                top_drivers TEXT,
                model_version TEXT,
                scored_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (customer_id)
                    REFERENCES customers(customer_id)
            )
        """)


if __name__ == "__main__":
    initialize_database()
    print(f"Database initialized: {DB_PATH}")
