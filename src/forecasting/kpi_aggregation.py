import sqlite3
import pandas as pd

DB_PATH = "database/finance.db"

# Load data
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM transactions", conn)
conn.close()

# Convert date
df["transaction_date"] = pd.to_datetime(df["transaction_date"])
df["date"] = df["transaction_date"].dt.date

# Revenue & Expense
revenue = df[df["transaction_type"] == "credit"]["amount"].sum()
expense = df[df["transaction_type"] == "debit"]["amount"].sum()
net_cash_flow = revenue - expense

# Daily aggregation
daily_kpis = df.groupby("date").agg(
    total_revenue=pd.NamedAgg(
        column="amount",
        aggfunc=lambda x: x[df.loc[x.index, "transaction_type"] == "credit"].sum()
    ),
    total_expense=pd.NamedAgg(
        column="amount",
        aggfunc=lambda x: x[df.loc[x.index, "transaction_type"] == "debit"].sum()
    ),
    avg_transaction_value=("amount", "mean"),
    anomaly_count=("is_anomaly", "sum")
).reset_index()

print("📊 KPI SUMMARY")
print(f"Total Revenue: {revenue:,.2f}")
print(f"Total Expense: {expense:,.2f}")
print(f"Net Cash Flow: {net_cash_flow:,.2f}")
print("\n📈 Sample Daily KPIs:")
print(daily_kpis.head())
