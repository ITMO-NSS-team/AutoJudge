import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import json

# db config
DB_NAME = "maseval"
DB_USER = "postgres"
DB_PASSWORD = ""  # set to None if no password
DB_HOST = "localhost"
DB_PORT = 5432


def drop_table():
    conn = get_conn(DB_NAME)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS who_when;")
    conn.commit()

    cur.close()
    conn.close()

    print("who_when dropped")


def get_conn(db):
    return psycopg2.connect(
        dbname=db, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )


def create_database():
    conn = get_conn("postgres")
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
    if not cur.fetchone():
        cur.execute(f"CREATE DATABASE {DB_NAME}")
        print(f"DB {DB_NAME} created")
    else:
        print(f"DB {DB_NAME} already exists")

    cur.close()
    conn.close()


def create_table():
    conn = get_conn(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS who_when (
            id TEXT NOT NULL,
            state_id TEXT PRIMARY KEY,
            state_index INT NOT NULL,
            content JSONB NOT NULL
        );
    """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_who_when_id ON who_when(id);")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_who_when_state_index ON who_when(state_index);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_who_when_content ON who_when USING GIN(content);"
    )

    conn.commit()
    cur.close()
    conn.close()


def load_data(fname: str):
    path = f"hf://datasets/Kevin355/Who_and_When/{fname}"
    return pd.read_parquet(path)


def transform(df):
    rows = []

    for _, r in df.iterrows():
        trace_id = r["question_ID"]
        history = r["history"]

        for i, state in enumerate(history, 1):
            state_id = f"{trace_id}_{i}"

            rows.append((trace_id, state_id, i, json.dumps(state)))

    return rows


def insert_rows(rows):
    if not rows:
        print("no rows")
        return

    conn = get_conn(DB_NAME)
    cur = conn.cursor()

    q = """
        INSERT INTO who_when (id, state_id, state_index, content)
        VALUES %s
        ON CONFLICT (state_id) DO NOTHING;
    """

    execute_values(cur, q, rows, page_size=1000)

    conn.commit()
    cur.close()
    conn.close()


def load_and_insert(fname: str):
    print(f"loading {fname} ...")
    df = load_data(fname)

    print("transform...")
    rows = transform(df)

    print(f"inserting {len(rows)} rows")
    insert_rows(rows)


def main():
    create_database()
    create_table()

    load_and_insert("Hand-Crafted.parquet")
    load_and_insert("Algorithm-Generated.parquet")

    print("done")


if __name__ == "__main__":
    drop_table()
    main()
