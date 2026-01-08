import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

DB_PATH = "database/finance.db"

# -----------------------------
# Load data
# -----------------------------
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM transactions", conn)
conn.close()

# -----------------------------
# Feature Engineering
# -----------------------------
df["log_amount"] = np.log1p(df["amount"])

# Transaction frequency per account
df["txn_count"] = df.groupby("account_id")["amount"].transform("count")

# Rolling z-score (behavioral anomaly)
df["rolling_mean"] = df.groupby("account_id")["amount"].transform(
    lambda x: x.rolling(10, min_periods=1).mean()
)
df["rolling_std"] = df.groupby("account_id")["amount"].transform(
    lambda x: x.rolling(10, min_periods=1).std()
)
df["rolling_zscore"] = (df["amount"] - df["rolling_mean"]) / df["rolling_std"]
df["rolling_zscore"] = df["rolling_zscore"].fillna(0)

# Time-based feature
df["transaction_date"] = pd.to_datetime(df["transaction_date"])
df["hour"] = df["transaction_date"].dt.hour
# Convert hour to cyclical features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)


# -----------------------------
# Isolation Forest (High Precision Mode)
# -----------------------------
features = df[["log_amount", "txn_count", "rolling_zscore", "hour_sin", "hour_cos"]]
 # strongest signal only
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

model = IsolationForest(
    n_estimators=300,       # slightly higher for stability
    contamination=0.02,     # match synthetic 2% anomalies
    random_state=42
)

# Train model and predict anomalies in one step
df["predicted_anomaly"] = model.fit_predict(features_scaled)

# Map the output to 0/1 (IsolationForest gives 1 for normal, -1 for anomaly)
df["predicted_anomaly"] = df["predicted_anomaly"].map({1: 0, -1: 1})
# -----------------------------
# Evaluation
# -----------------------------
precision = precision_score(df["is_anomaly"], df["predicted_anomaly"])
cm = confusion_matrix(df["is_anomaly"], df["predicted_anomaly"])

print(f"🎯 Anomaly Detection Precision: {precision:.2f}")
print("Confusion Matrix:")
print(cm)
