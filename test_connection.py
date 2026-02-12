
import psycopg2

try:
    conn = psycopg2.connect(
        dbname="Lloan_app;",       # your database name
        user="postgres",         # your PostgreSQL username
        password="#erisimasol1985", # the password you set for that user
        host="localhost",
        port="5432"
    )
    print("✅ Connection successful!")
    conn.close()
except Exception as e:
    print("❌ Error:", e)

