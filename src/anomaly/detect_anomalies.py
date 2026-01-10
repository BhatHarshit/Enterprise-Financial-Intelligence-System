import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

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

# Cyclical encoding for hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# -----------------------------
# Prepare features
# -----------------------------
features = df[["log_amount", "txn_count", "rolling_zscore", "hour_sin", "hour_cos"]]

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# -----------------------------
# Isolation Forest (High Precision)
# -----------------------------
model = IsolationForest(
    n_estimators=300,
    contamination=0.02,  # 2% anomalies in synthetic data
    random_state=42,
    max_features=3       # optional: subset of features per tree for stability
)

# Train model
model.fit(features_scaled)

# -----------------------------
# Score-based anomaly detection
# -----------------------------
# Lower scores are more anomalous
df["anomaly_score"] = model.score_samples(features_scaled)

# Choose top X% anomalies based on score
threshold = df["anomaly_score"].quantile(0.02)  # tune between 0.01–0.03
df["predicted_anomaly"] = (df["anomaly_score"] <= threshold).astype(int)

# -----------------------------
# Evaluation Metrics
# -----------------------------
precision = precision_score(df["is_anomaly"], df["predicted_anomaly"])
recall = recall_score(df["is_anomaly"], df["predicted_anomaly"])
f1 = f1_score(df["is_anomaly"], df["predicted_anomaly"])
cm = confusion_matrix(df["is_anomaly"], df["predicted_anomaly"])

print("🎯 Anomaly Detection Metrics")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-Score:  {f1:.2f}")
print("Confusion Matrix:")
print(cm)
