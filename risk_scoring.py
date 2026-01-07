import pandas as pd


def score_risk(df, model, scaler, features):
df = df.copy()
df["delay_risk_score"] = model.predict_proba(
scaler.transform(df[features])
)[:, 1]


high_risk = df[df["delay_risk_score"] > 0.7]
high_risk.to_csv("high_risk_supply_chain_orders.csv", index=False)


return df
