import streamlit as st
import pandas as pd
import joblib
import psycopg2
from psycopg2 import sql
from datetime import datetime

# Load trained model and training columns
model, training_columns = joblib.load("loan_model.pkl")

# Database connection function
def get_connection():
    return psycopg2.connect(
        dbname="Lloan_app",             # fixed typo
        user="postgres",               # change to your DB user
        password="#erisimasol1985",    # change to your DB password
        host="localhost",
        port="5432"
    )

# Create tables if not exist
def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS applicants (
        id SERIAL PRIMARY KEY,
        age INT,
        income NUMERIC,
        employment_status VARCHAR(50),
        loan_amount NUMERIC,
        loan_term INT,
        credit_score INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        applicant_id INT REFERENCES applicants(id),
        prediction VARCHAR(20),
        probability NUMERIC,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    cursor.close()
    conn.close()

# Initialize DB
init_db()

# Streamlit UI
st.title("Loan Approval Prediction App")

age = st.number_input("Age", min_value=18, max_value=100)
income = st.number_input("Monthly Income", min_value=0)
employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
loan_amount = st.number_input("Loan Amount", min_value=1000)
loan_term = st.number_input("Loan Term (months)", min_value=6)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850)

if st.button("Predict"):
    # Save applicant data
    conn = get_connection()
    cursor = conn.cursor()
    created_at = datetime.now()
    cursor.execute("""
        INSERT INTO applicants (age, income, employment_status, loan_amount, loan_term, credit_score, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
    """, (age, income, employment_status, loan_amount, loan_term, credit_score, created_at))
    applicant_id = cursor.fetchone()[0]

    # Prepare input for model
    input_data = pd.DataFrame({
        "Age": [age],
        "Income": [income],
        "EmploymentStatus": [employment_status],
        "LoanAmount": [loan_amount],
        "LoanTerm": [loan_term],
        "CreditScore": [credit_score]
    })

    # One-hot encode applicant input
    input_data = pd.get_dummies(input_data)

    # Add missing columns with 0
    for col in training_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match training
    input_data = input_data[training_columns]

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Save prediction
    cursor.execute("""
    INSERT INTO predictions (applicant_id, prediction, probability, created_at)
    VALUES (%s, %s, %s, %s);
""", (
    applicant_id,
    "Approved" if prediction == 1 else "Rejected",
    float(probability),   # ✅ convert NumPy float to Python float
    created_at
))

    conn.commit()
    cursor.close()
    conn.close()

    st.success(f"Prediction: {'Approved' if prediction==1 else 'Rejected'}")
    st.info(f"Approval Probability: {probability:.2f}")
