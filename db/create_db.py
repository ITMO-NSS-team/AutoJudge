import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import json

# CONFIG
DB_NAME = "maseval"
DB_USER = "postgres"
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = 5432

def drop_table():
    """Deletes the who_when table if it exists. Use with caution!"""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()
        
        cur.execute("DROP TABLE IF EXISTS who_when;")
        conn.commit()
        print("Table who_when dropped successfully.")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error dropping table: {e}")

def create_database():
    conn = psycopg2.connect(
        dbname="postgres",
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
    exists = cur.fetchone()

    if not exists:
        cur.execute(f"CREATE DATABASE {DB_NAME}")
        print(f"Database {DB_NAME} created.")
    else:
        print(f"Database {DB_NAME} already exists.")

    cur.close()
    conn.close()


# 2. CREATE TABLE who_when
def create_table():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS who_when (
            id TEXT NOT NULL,
            state_id TEXT PRIMARY KEY,
            state_index INTEGER NOT NULL,
            content JSONB NOT NULL
        );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_who_when_id ON who_when(id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_who_when_state_index ON who_when(state_index);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_who_when_content ON who_when USING GIN(content);")

    conn.commit()
    cur.close()
    conn.close()


# 3. load dataset from Hugging Face
def load_data():
    df = pd.read_parquet(
        "hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet"
    )
    return df


# 4. TRANSFORM
def transform(df):
    rows = []

    for _, row in df.iterrows():
        trace_id = row["question_ID"]
        trace_uuid = trace_id

        history = row["history"]

        for i, state_dict in enumerate(history, start=1):
            state_id = f"{trace_id}_{i}"

            rows.append((
                trace_uuid,
                state_id,
                i,
                json.dumps(state_dict)
            ))

    return rows


# 5. INSERT
def insert_rows(rows):
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()

    query = """
        INSERT INTO who_when (id, state_id, state_index, content)
        VALUES %s
        ON CONFLICT (state_id) DO NOTHING;
    """

    execute_values(cur, query, rows, page_size=1000)

    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    # drop_table()
    create_database()
    create_table()

    print("Loading parquet...")
    df = load_data()

    print("Transforming...")
    rows = transform(df)

    print(f"Inserting {len(rows)} rows...")
    insert_rows(rows)

    print("Done")
