import pandas as pd


def preprocess_data(df):
df = df.copy()


df["inventory_coverage_days"] = df["inventory_level"] / df["daily_demand"]
df["total_lead_time"] = df["supplier_lead_time"] + df["transportation_time"]


features = [
"supplier_lead_time",
"transportation_time",
"inventory_level",
"daily_demand",
"supplier_reliability",
"warehouse_utilization",
"inventory_coverage_days",
"total_lead_time"
]


X = df[features]
y = df["delay_risk"]


return X, y, df
