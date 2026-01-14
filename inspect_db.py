import sqlite3
import pandas as pd

conn = sqlite3.connect("database/finance.db")

tables = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table';",
    conn
)["name"].tolist()

for table in tables:
    print(f"\n📊 TABLE: {table}")
    print("-" * 50)

    df = pd.read_sql(f"SELECT * FROM {table} LIMIT 1", conn)

    print(f"Columns ({len(df.columns)}):")
    for col in df.columns:
        print(" -", col)

    print("\nData Types:")
    print(df.dtypes)

    count = pd.read_sql(
        f"SELECT COUNT(*) AS cnt FROM {table}",
        conn
    )["cnt"][0]

    print(f"\nRows: {count}")

conn.close()
