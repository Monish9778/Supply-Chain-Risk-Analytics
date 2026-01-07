from data_generation import generate_supply_chain_data
from preprocessing import preprocess_data
from model_training import train_model
from risk_scoring import score_risk


# Generate data
df = generate_supply_chain_data()


# Preprocess
X, y, df = preprocess_data(df)


# Train model
model, scaler = train_model(X, y)


# Feature list
features = X.columns.tolist()


# Risk scoring
final_df = score_risk(df, model, scaler, features)


print("\nSupply Chain Risk Analytics executed successfully")
