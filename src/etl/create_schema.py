import sqlite3

DB_PATH = "database/finance.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    transaction_date TIMESTAMP,
    account_id TEXT,
    transaction_type TEXT,
    amount REAL,
    merchant TEXT,
    category TEXT,
    region TEXT,
    channel TEXT,
    is_anomaly INTEGER
)
""")

conn.commit()
conn.close()

print("✅ Database schema created successfully")
