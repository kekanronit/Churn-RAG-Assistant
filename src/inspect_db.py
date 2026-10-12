
import sqlite3
from pathlib import Path

db_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "churn.db"
)

with sqlite3.connect(db_path) as conn:
    print("Database:", db_path)

    print("\nCUSTOMERS TABLE:")
    columns = conn.execute(
        "PRAGMA table_info(customers)"
    ).fetchall()

    for column in columns:
        print(column)

    print("\nSUPPORT TICKETS FOREIGN KEY:")
    foreign_keys = conn.execute(
        "PRAGMA foreign_key_list(support_tickets)"
    ).fetchall()

    for foreign_key in foreign_keys:
        print(foreign_key)

    print("\nCUSTOMER COUNT:")
    print(
        conn.execute(
            "SELECT COUNT(*) FROM customers"
        ).fetchone()[0]
    )
