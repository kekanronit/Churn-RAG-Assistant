
import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "churn.db"


def repair_database():
    conn = sqlite3.connect(DB_PATH)

    try:
        # Foreign-key enforcement must be disabled
        # before starting the repair transaction.
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN")

        # Preserve and repair the customer records.
        conn.execute("""
            CREATE TABLE customers_new (
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
            INSERT INTO customers_new
            SELECT customer_id, gender, senior_citizen,
                   partner, dependents, tenure_months,
                   contract, payment_method, paperless_billing,
                   internet_service, monthly_charges,
                   total_charges, churn_label
            FROM customers
        """)

        # Remove dependent tables before replacing customers.
        conn.execute("DROP TABLE IF EXISTS support_tickets")
        conn.execute("DROP TABLE IF EXISTS predictions")
        conn.execute("DROP TABLE customers")

        conn.execute("""
            ALTER TABLE customers_new RENAME TO customers
        """)

        # Recreate the dependent tables.
        conn.execute("""
            CREATE TABLE support_tickets (
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
            CREATE TABLE predictions (
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

        count = conn.execute(
            "SELECT COUNT(*) FROM customers"
        ).fetchone()[0]

        conn.commit()

        # Re-enable foreign-key enforcement.
        conn.execute("PRAGMA foreign_keys = ON")

        violations = conn.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

        print("Customer records preserved:", count)
        print("Foreign-key violations:", violations)
        print("Database repair completed.")

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    repair_database()
