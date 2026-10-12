
from db import get_connection

with get_connection() as conn:
    rows = conn.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """).fetchall()

    print("Tables in the database:")

    for row in rows:
        print("-", row[0])

