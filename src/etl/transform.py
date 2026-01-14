"""
Enterprise-grade Transform Layer
- Schema validation
- Type enforcement
- No semantic distortion
"""

import pandas as pd


REQUIRED_COLUMNS = {
    "transaction_id",
    "transaction_date",
    "account_id",
    "transaction_type",
    "amount",
    "merchant",
    "category",
    "region",
    "channel",
    "is_anomaly",
}


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    # -------------------------------
    # Normalize column names
    # -------------------------------
    df.columns = df.columns.str.lower().str.strip()

    # -------------------------------
    # Schema validation (FAIL FAST)
    # -------------------------------
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")

    # -------------------------------
    # Type enforcement
    # -------------------------------
    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"], errors="coerce"
    )

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["is_anomaly"] = pd.to_numeric(df["is_anomaly"], errors="coerce").fillna(0).astype(int)

    # -------------------------------
    # Drop invalid rows
    # -------------------------------
    df = df.dropna(
        subset=[
            "transaction_id",
            "transaction_date",
            "account_id",
            "amount",
        ]
    )

    # -------------------------------
    # Remove duplicates (idempotent ETL)
    # -------------------------------
    df = df.drop_duplicates(subset=["transaction_id"])

    return df
