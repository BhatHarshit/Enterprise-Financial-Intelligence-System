"""
Enterprise ETL Pipeline
- Schema-aligned with generated financial data
- SQLite-safe
"""

import pandas as pd
import sqlite3
from pathlib import Path
from .transform import transform_data


# -------------------------------
# CONFIG
# -------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "src" / "data" / "transactions.csv"
DB_PATH = BASE_DIR / "database" / "finance.db"


# -------------------------------
# EXTRACT
# -------------------------------

def extract_data():
    return pd.read_csv(DATA_PATH)


# -------------------------------
# LOAD
# -------------------------------

def load_to_db(df: pd.DataFrame):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # -------------------------------
    # Clear the table first to prevent UNIQUE constraint errors
    # -------------------------------
    cur.execute("DELETE FROM transactions;")
    conn.commit()

    # -------------------------------
    # Insert new clean data
    # -------------------------------
    df.to_sql(
        "transactions",
        conn,
        if_exists="append",
        index=False,
        chunksize=500,      # ✅ prevents SQL variable overflow
        method=None
    )

    conn.commit()
    conn.close()


# -------------------------------
# RUNNER
# -------------------------------

def run_etl():
    print("🚀 Starting ETL pipeline")

    df_raw = extract_data()
    print(f"📥 Extracted {len(df_raw)} records")

    df_clean = transform_data(df_raw)
    print(f"🧹 Cleaned to {len(df_clean)} records")

    load_to_db(df_clean)
    print("✅ Loaded into SQLite successfully")

    print("🎯 ETL completed")


if __name__ == "__main__":
    run_etl()
