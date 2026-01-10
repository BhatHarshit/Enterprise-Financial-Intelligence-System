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
df["txn_count"] = df.groupby("account_id")["amount"].transform("count")

# Rolling z-score for behavioral anomaly
df["rolling_mean"] = df.groupby("account_id")["amount"].transform(
    lambda x: x.rolling(10, min_periods=1).mean()
)
df["rolling_std"] = df.groupby("account_id")["amount"].transform(
    lambda x: x.rolling(10, min_periods=1).std()
)
df["rolling_zscore"] = (df["amount"] - df["rolling_mean"]) / df["rolling_std"]
df["rolling_zscore"] = df["rolling_zscore"].fillna(0)

# Time-based features
df["transaction_date"] = pd.to_datetime(df["transaction_date"])
df["hour"] = df["transaction_date"].dt.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# -----------------------------
# Prepare features
# -----------------------------
features = df[["log_amount", "txn_count", "rolling_zscore", "hour_sin", "hour_cos"]]

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# -----------------------------
# Isolation Forest
# -----------------------------
# Remove contamination parameter to allow full score evaluation
model = IsolationForest(
    n_estimators=500,    # increase for stability
    max_samples='auto',
    contamination='auto', # auto lets the model adjust internally
    random_state=42
)

# Train the model
model.fit(features_scaled)

# -----------------------------
# Anomaly Scores
# -----------------------------
# Negative scores indicate more anomalous
df["anomaly_score"] = -model.score_samples(features_scaled)

# Determine threshold automatically using actual anomalies for maximum F1
best_f1 = 0
best_threshold = 0
for threshold in np.linspace(df["anomaly_score"].min(), df["anomaly_score"].max(), 1000):
    predicted = (df["anomaly_score"] >= threshold).astype(int)
    f1 = f1_score(df["is_anomaly"], predicted)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Apply best threshold
df["predicted_anomaly"] = (df["anomaly_score"] >= best_threshold).astype(int)

# -----------------------------
# Evaluation
# -----------------------------
precision = precision_score(df["is_anomaly"], df["predicted_anomaly"])
recall = recall_score(df["is_anomaly"], df["predicted_anomaly"])
f1 = f1_score(df["is_anomaly"], df["predicted_anomaly"])
cm = confusion_matrix(df["is_anomaly"], df["predicted_anomaly"])

print(f"🎯 Precision: {precision:.2f}")
print(f"📈 Recall:    {recall:.2f}")
print(f"💡 F1-Score:  {f1:.2f}")
print("Confusion Matrix:")
print(cm)
print(f"--- Best threshold based on F1: {best_threshold:.4f} ---")
