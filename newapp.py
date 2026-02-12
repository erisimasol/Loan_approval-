import psycopg2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Connect to PostgreSQL
def get_connection():
    return psycopg2.connect(
        dbname="loan_app",
        user="postgres",
        password="yourpassword",
        host="localhost",
        port="5432"
    )

# Insert applicant data manually
def insert_applicant(age, income, employment_status, loan_amount, loan_term, credit_score, loan_status):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO applicants (age, income, employment_status, loan_amount, loan_term, credit_score, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, NOW()) RETURNING id;
    """, (age, income, employment_status, loan_amount, loan_term, credit_score))
    applicant_id = cursor.fetchone()[0]

    # Store true label for training
    cursor.execute("""
        INSERT INTO predictions (applicant_id, prediction, probability, created_at)
        VALUES (%s, %s, %s, NOW());
    """, (applicant_id, loan_status, None))
    conn.commit()
    cursor.close()
    conn.close()

# Train model from DB
def train_model():
    conn = get_connection()
    df = pd.read_sql("SELECT a.*, p.prediction AS Loan_Status FROM applicants a JOIN predictions p ON a.id=p.applicant_id", conn)
    conn.close()

    X = df[["age","income","employment_status","loan_amount","loan_term","credit_score"]]
    y = df["Loan_Status"]

    # Encode categorical
    X = pd.get_dummies(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, "loan_model.pkl")
    print("Model trained and saved as loan_model.pkl")

# Predict new applicant
def predict_applicant(age, income, employment_status, loan_amount, loan_term, credit_score):
    model = joblib.load("loan_model.pkl")
    input_data = pd.DataFrame({
        "age":[age],
        "income":[income],
        "employment_status":[employment_status],
        "loan_amount":[loan_amount],
        "loan_term":[loan_term],
        "credit_score":[credit_score]
    })
    input_data = pd.get_dummies(input_data)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    print(f"Prediction: {prediction}, Probability: {probability:.2f}")
