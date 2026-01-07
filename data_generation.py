import numpy as np
import pandas as pd


np.random.seed(42)


def generate_supply_chain_data(n_rows=5000):
data = pd.DataFrame({
"supplier_lead_time": np.random.normal(12, 4, n_rows).clip(1),
"transportation_time": np.random.normal(7, 2, n_rows).clip(1),
"inventory_level": np.random.normal(500, 150, n_rows).clip(50),
"daily_demand": np.random.normal(60, 15, n_rows).clip(10),
"order_quantity": np.random.normal(300, 80, n_rows).clip(50),
"supplier_reliability": np.random.uniform(0.7, 1.0, n_rows),
"warehouse_utilization": np.random.uniform(0.5, 0.95, n_rows)
})


delay_probability = (
(data["supplier_lead_time"] > 14).astype(int) * 0.25 +
(data["transportation_time"] > 9).astype(int) * 0.25 +
(data["inventory_level"] < 300).astype(int) * 0.30 +
(data["supplier_reliability"] < 0.85).astype(int) * 0.20
)


data["delay_risk"] = (delay_probability > 0.45).astype(int)
return data
