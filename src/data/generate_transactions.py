import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)

N_TRANSACTIONS = 50000

start_date = datetime(2022, 1, 1)

data = []

categories = ["Retail", "Food", "Travel", "Utilities", "Healthcare", "Entertainment"]
merchants = ["Amazon", "Walmart", "Uber", "Zomato", "Netflix", "Airbnb"]
regions = ["North", "South", "East", "West"]
channels = ["Online", "POS", "Bank"]

for i in range(N_TRANSACTIONS):
    txn_date = start_date + timedelta(days=random.randint(0, 900))
    amount = round(np.random.normal(2000, 800), 2)

    is_anomaly = 0
    if random.random() < 0.02:  # 2% anomalies
        amount *= random.randint(5, 10)
        is_anomaly = 1

    data.append([
        f"TXN{i+1}",
        txn_date,
        f"ACC{random.randint(1000, 5000)}",
        random.choice(["credit", "debit"]),
        abs(amount),
        random.choice(merchants),
        random.choice(categories),
        random.choice(regions),
        random.choice(channels),
        is_anomaly
    ])

columns = [
    "transaction_id",
    "transaction_date",
    "account_id",
    "transaction_type",
    "amount",
    "merchant",
    "category",
    "region",
    "channel",
    "is_anomaly"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("transactions.csv", index=False)

print("✅ Generated 50,000 financial transactions")
