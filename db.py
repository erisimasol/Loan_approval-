import psycopg2

conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="yourpassword",
    host="localhost",
    port="5432"
)
conn.autocommit = True
cursor = conn.cursor()

cursor.execute("CREATE DATABASE loan_app;")
conn.close()

# Connect to loan_app
conn = psycopg2.connect(
    dbname="loan_app",
    user="postgres",
    password="yourpassword",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE applicants (
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
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    applicant_id INT REFERENCES applicants(id),
    prediction VARCHAR(20),
    probability NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

conn.commit()
conn.close()
