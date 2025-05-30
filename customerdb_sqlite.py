import sqlite3

# Create SQLite DB with a customers table
conn = sqlite3.connect("customers.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    state TEXT
)
""")

cursor.executemany("""
INSERT INTO customers (name, email, state)
VALUES (?, ?, ?)
""", [
    ("Alice", "alice@example.com", "California"),
    ("Bob", "bob@example.com", "Texas"),
    ("Charlie", "charlie@example.com", "California"),
])

conn.commit()
conn.close()

