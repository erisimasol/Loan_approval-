import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.utils import resample

# -----------------------------
# Step 1: Generate synthetic dataset
# -----------------------------
np.random.seed(42)
n_samples = 1500

ages = np.random.randint(18, 65, n_samples)
incomes = np.random.randint(1000, 30000, n_samples)
employment_statuses = np.random.choice(["Employed", "Self-Employed", "Unemployed"], n_samples)
loan_amounts = np.random.randint(1000, 50000, n_samples)
loan_terms = np.random.choice([12, 24, 36, 48, 60], n_samples)
credit_scores = np.random.randint(300, 850, n_samples)

loan_statuses = []
for income, credit, emp, loan in zip(incomes, credit_scores, employment_statuses, loan_amounts):
    if emp == "Unemployed" and credit < 600:
        loan_statuses.append("Rejected")
    elif income > 15000 and credit > 650:
        loan_statuses.append("Approved")
    elif emp == "Self-Employed" and income > 12000 and credit > 680:
        loan_statuses.append("Approved")
    elif income > 8000 and credit > 700 and loan < 30000:
        loan_statuses.append("Approved")
    else:
        loan_statuses.append("Rejected")

data = pd.DataFrame({
    "Age": ages,
    "Income": incomes,
    "EmploymentStatus": employment_statuses,
    "LoanAmount": loan_amounts,
    "LoanTerm": loan_terms,
    "CreditScore": credit_scores,
    "Loan_Status": loan_statuses
})

print("Dataset created with shape:", data.shape)
print(data["Loan_Status"].value_counts())

# -----------------------------
# Step 2: Balance dataset
# -----------------------------
approved = data[data.Loan_Status == "Approved"]
rejected = data[data.Loan_Status == "Rejected"]

if len(approved) < len(rejected):
    approved_upsampled = resample(approved, replace=True, n_samples=len(rejected), random_state=42)
    data_balanced = pd.concat([approved_upsampled, rejected])
else:
    rejected_upsampled = resample(rejected, replace=True, n_samples=len(approved), random_state=42)
    data_balanced = pd.concat([approved, rejected_upsampled])

print("Balanced dataset shape:", data_balanced.shape)
print(data_balanced["Loan_Status"].value_counts())

# -----------------------------
# Step 3: Train XGBoost model
# -----------------------------
X = pd.get_dummies(data_balanced.drop("Loan_Status", axis=1))
y = (data_balanced["Loan_Status"] == "Approved").astype(int)

model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X, y)

# -----------------------------
# Step 4: Save model + training columns
# -----------------------------
model.save_model("loan_model.json")  # native XGBoost format
pd.Series(X.columns).to_json("loan_features.json")  # save feature names

print("✅ XGBoost model trained and saved with feature columns")
