from flask import Flask, request, jsonify
import psycopg2
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)
model = joblib.load("loan_model.pkl")

def get_connection():
    return psycopg2.connect(
        dbname="loan_app",
        user="postgres",
        password="yourpassword",
        host="localhost",
        port="5432"
    )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    conn = get_connection()
    cursor = conn.cursor()
    created_at = datetime.now()

    cursor.execute("""
        INSERT INTO applicants (age, income, employment_status, loan_amount, loan_term, credit_score, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
    """, (data["Age"], data["Income"], data["EmploymentStatus"], data["LoanAmount"], data["LoanTerm"], data["CreditScore"], created_at))
    applicant_id = cursor.fetchone()[0]

    cursor.execute("""
        INSERT INTO predictions (applicant_id, prediction, probability, created_at)
        VALUES (%s, %s, %s, %s);
    """, (applicant_id, "Approved" if prediction==1 else "Rejected", probability, created_at))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({
        "prediction": "Approved" if prediction==1 else "Rejected",
        "probability": round(probability, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
