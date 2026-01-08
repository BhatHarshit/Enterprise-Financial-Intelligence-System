import sqlite3
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error

DB_PATH = "database/finance.db"

# Load data
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql(
    "SELECT transaction_date, transaction_type, amount FROM transactions",
    conn
)
conn.close()

df["transaction_date"] = pd.to_datetime(df["transaction_date"])
df["date"] = df["transaction_date"].dt.date

# Daily cash flow (enterprise KPI)
daily = (
    df.groupby(["date", "transaction_type"])["amount"]
    .sum()
    .unstack(fill_value=0)
)

daily["net_cash_flow"] = daily.get("credit", 0) - daily.get("debit", 0)

ts = daily["net_cash_flow"]

# Train-test split
train_size = int(len(ts) * 0.8)
train, test = ts.iloc[:train_size], ts.iloc[train_size:]

# Baseline forecast (mean of last 7 days)
baseline_forecast = [train.tail(7).mean()] * len(test)

# ETS with weekly seasonality
ets_model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="add",
    seasonal_periods=7
).fit()

ets_forecast = ets_model.forecast(len(test))

# Evaluation
baseline_mape = mean_absolute_percentage_error(test, baseline_forecast)
ets_mape = mean_absolute_percentage_error(test, ets_forecast)

improvement = (baseline_mape - ets_mape) / baseline_mape * 100

print("📈 Forecasting Results (Net Cash Flow)")
print(f"Baseline MAPE: {baseline_mape:.2f}")
print(f"ETS MAPE: {ets_mape:.2f}")
print(f"Accuracy Improvement: {improvement:.2f}%")
