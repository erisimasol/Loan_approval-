import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Example dataset (replace with your own loan data CSV)
data = pd.read_csv("loan_data.csv")

# Features and target
X = data.drop("Loan_Status", axis=1)   # Loan_Status = Approved/Rejected
y = data["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "loan_model.pkl")

print("Model trained and saved as loan_model.pkl")
