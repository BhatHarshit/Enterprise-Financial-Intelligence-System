import pandas as pd
import sqlite3

DB_PATH = "database/finance.db"
CSV_PATH = "src/data/transactions.csv"

df = pd.read_csv(CSV_PATH)

conn = sqlite3.connect(DB_PATH)

df.to_sql(
    name="transactions",
    con=conn,
    if_exists="append",
    index=False
)

conn.close()

print(f"✅ Loaded {len(df)} transactions into SQL database")
