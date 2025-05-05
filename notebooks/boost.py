import cudf
import cuml
from cuml.model_selection import train_test_split
import xgboost as xgb
import numpy as np

# Load your dataset into a GPU dataframe (cudf)
df = cudf.read_csv("contract_data.csv")

# Assume df has these columns:
# 'customer_age', 'asset_value', 'credit_score', ..., 'time_to_default', 'defaulted'

# Step 1: Feature Engineering (assuming basic features exist)
features = [
    "customer_age",
    "asset_value",
    "credit_score",
    "contract_length",
    "interest_rate",
    "payment_frequency",
]

X = df[features]

# For Survival Analysis:
# 'time_to_default' -> Time until event or censoring (numeric)
# 'defaulted' -> Event occurrence indicator (1 if defaulted, 0 if censored)
y_time = df["time_to_default"].values
y_event = df["defaulted"].values

# XGBoost survival analysis expects the label as time with event indicator
# Prepare DMatrix for XGBoost with survival data
dtrain = xgb.DMatrix(X, label=y_time)
dtrain.set_float_info(
    "label_lower_bound", y_event
)  # 1 if event occurred, 0 if censored
dtrain.set_float_info("label_upper_bound", np.where(y_event, y_time, np.inf))

# Step 2: Train-test split
# Fix the train_test_split call by combining the targets into arrays
import cupy as cp

combined_y = cp.column_stack((y_time, y_event))
X_train, X_test, y_combined_train, y_combined_test = train_test_split(
    X, combined_y, test_size=0.2, random_state=42
)

# Split the combined target back into time and event
y_time_train = y_combined_train[:, 0]
y_event_train = y_combined_train[:, 1]
y_time_test = y_combined_test[:, 0]
y_event_test = y_combined_test[:, 1]

# Create DMatrix for train and test separately
dtrain = xgb.DMatrix(X_train, label=y_time_train)
dtrain.set_float_info("label_lower_bound", y_event_train)
dtrain.set_float_info(
    "label_upper_bound", np.where(y_event_train, y_time_train, np.inf)
)

dtest = xgb.DMatrix(X_test, label=y_time_test)
dtest.set_float_info("label_lower_bound", y_event_test)
dtest.set_float_info("label_upper_bound", np.where(y_event_test, y_time_test, np.inf))

# Step 3: Define XGBoost parameters (Survival Analysis - Cox PH)
params = {
    "objective": "survival:cox",
    "eval_metric": "cox-nloglik",
    "tree_method": "hist",  # Changed from gpu_hist
    "device": "cuda",  # Added device parameter
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

# Step 4: Train the model
num_round = 100
evals = [(dtrain, "train"), (dtest, "test")]
bst = xgb.train(params, dtrain, num_round, evals, early_stopping_rounds=10)

# Step 5: Predict risk scores (higher scores indicate higher risk)
risk_scores = bst.predict(dtest)

# Step 6: Evaluate using Concordance Index (C-index)
# First install lifelines if you haven't:
# pip install lifelines

# Replace the existing concordance index calculation with:
from lifelines.utils import concordance_index

# Convert GPU arrays to CPU for concordance calculation
y_time_test_cpu = y_time_test.get() if hasattr(y_time_test, "get") else y_time_test
y_event_test_cpu = y_event_test.get() if hasattr(y_event_test, "get") else y_event_test
risk_scores_cpu = risk_scores.get() if hasattr(risk_scores, "get") else risk_scores

# Calculate concordance index
c_index = concordance_index(y_time_test_cpu, -risk_scores_cpu, y_event_test_cpu)
print(f"C-index on test data: {c_index:.4f}")

# Interpretation:
# Higher C-index (close to 1.0) indicates excellent predictive ability.
