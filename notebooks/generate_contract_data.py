import numpy as np
import pandas as pd
import datetime as dt

# Set random seed for reproducibility
np.random.seed(42)

# Number of records
n_samples = 10000

# Generate synthetic data
data = {
    "customer_age": np.random.normal(45, 15, n_samples).clip(18, 85),
    "asset_value": np.random.lognormal(11, 1, n_samples),  # in dollars
    "credit_score": np.random.normal(700, 100, n_samples).clip(300, 850),
    "contract_length": np.random.choice([12, 24, 36, 48, 60], n_samples),  # months
    "interest_rate": np.random.normal(0.05, 0.02, n_samples).clip(0.02, 0.15),
    "payment_frequency": np.random.choice(
        [1, 2, 4, 12], n_samples
    ),  # 1=annual, 12=monthly
}

# Generate survival data
# Higher risk for: older age, lower credit score, higher interest rate
risk_scores = (
    0.03 * data["customer_age"]
    + -0.005 * (data["credit_score"] - 500)
    + 20 * data["interest_rate"]
)
risk_scores = (risk_scores - risk_scores.min()) / (
    risk_scores.max() - risk_scores.min()
)

# Generate time to default based on risk scores
base_time = np.random.exponential(24, n_samples)  # base survival time (months)
time_to_default = (base_time / (risk_scores + 0.1)).clip(0, 60)

# Generate censoring (some contracts don't default within observation period)
censoring_time = np.random.uniform(0, 60, n_samples)
defaulted = (time_to_default <= censoring_time).astype(int)
observed_time = np.minimum(time_to_default, censoring_time)

data["time_to_default"] = observed_time
data["defaulted"] = defaulted

# Convert to DataFrame and round numeric values
df = pd.DataFrame(data)
df["customer_age"] = df["customer_age"].round(0)
df["asset_value"] = df["asset_value"].round(2)
df["credit_score"] = df["credit_score"].round(0)
df["interest_rate"] = df["interest_rate"].round(4)
df["time_to_default"] = df["time_to_default"].round(2)

# Save to CSV
df.to_csv("contract_data.csv", index=False)

print("Dataset summary:")
print("-" * 50)
print(f"Total records: {n_samples}")
print(f"Default rate: {defaulted.mean():.1%}")
print("\nFeature statistics:")
print(df.describe())
