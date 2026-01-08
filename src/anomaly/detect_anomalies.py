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

# -----------------------------
# Isolation Forest (High Precision Mode)
# -----------------------------
features = df[["log_amount"]]  # strongest signal only
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

model = IsolationForest(
    n_estimators=200,
    contamination=0.02,  # match synthetic 2% anomalies
    random_state=42
)
model.fit(features_scaled)

# Anomaly scores
df["anomaly_score"] = model.score_samples(features_scaled)

# Flag top 2% most anomalous
threshold = df["anomaly_score"].quantile(0.02)
df["predicted_anomaly"] = (df["anomaly_score"] <= threshold).astype(int)

# -----------------------------
# Evaluation
# -----------------------------
precision = precision_score(df["is_anomaly"], df["predicted_anomaly"])
cm = confusion_matrix(df["is_anomaly"], df["predicted_anomaly"])

print(f"🎯 Anomaly Detection Precision: {precision:.2f}")
print("Confusion Matrix:")
print(cm)
