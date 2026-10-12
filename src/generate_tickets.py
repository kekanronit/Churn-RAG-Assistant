
import random
import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "churn.db"

random.seed(42)

CATEGORIES = [
    "Billing",
    "Internet",
    "Technical Support",
    "Service Quality",
]


def generate_tickets():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        # Ensure the ticket table exists.
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

        customers = conn.execute("""
            SELECT customer_id, churn_label
            FROM customers
            ORDER BY customer_id
        """).fetchall()

        # Regenerate the synthetic data consistently on each run.
        conn.execute("DELETE FROM support_tickets")

        tickets = []
        ticket_id = 1

        for customer_id, churn_label in customers:
            # Synthetic assumption: churned customers are
            # somewhat more likely to have support tickets.
            probability = (
                0.35 if churn_label == "Yes" else 0.18
            )

            if random.random() < probability:
                number_of_tickets = random.randint(1, 3)

                for _ in range(number_of_tickets):
                    category = random.choice(CATEGORIES)
                    month = random.randint(1, 10)
                    day = random.randint(1, 28)
                    resolved = random.choice([0, 1])

                    tickets.append((
                        ticket_id,
                        customer_id,
                        category,
                        f"2026-{month:02d}-{day:02d}",
                        resolved,
                        f"Synthetic {category.lower()} ticket",
                    ))

                    ticket_id += 1

        conn.executemany("""
            INSERT INTO support_tickets (
                ticket_id, customer_id, category,
                created_at, resolved, notes
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, tickets)

        ticket_count = conn.execute("""
            SELECT COUNT(*) FROM support_tickets
        """).fetchone()[0]

        customer_count = conn.execute("""
            SELECT COUNT(DISTINCT customer_id)
            FROM support_tickets
        """).fetchone()[0]

        print(f"Customers checked: {len(customers)}")
        print(f"Synthetic tickets created: {ticket_count}")
        print(f"Customers with tickets: {customer_count}")


if __name__ == "__main__":
    generate_tickets()
