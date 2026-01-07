import sqlite3
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score

DB_PATH = "database/finance.db"

# Load data from SQL
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM transactions", conn)
conn.close()

# Feature engineering
features = df[["amount"]]

# Train Isolation Forest
model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

df["predicted_anomaly"] = model.fit_predict(features)
df["predicted_anomaly"] = df["predicted_anomaly"].map({1: 0, -1: 1})

# Precision calculation
precision = precision_score(df["is_anomaly"], df["predicted_anomaly"])

print(f"🎯 Anomaly Detection Precision: {precision:.2f}")
