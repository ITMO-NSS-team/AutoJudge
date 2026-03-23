import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import json

from dotenv import load_dotenv
import os

load_dotenv(".env")

# db config
DB_NAME = "maseval"
DB_USER = "postgres"
DB_PASSWORD = ""  # set to None if no password
DB_HOST = "localhost"
DB_PORT = 5432


def drop_table():
    conn = get_conn(DB_NAME)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS trail;")
    conn.commit()

    cur.close()
    conn.close()

    print("trail dropped")


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
        CREATE TABLE IF NOT EXISTS trail (
            id TEXT NOT NULL,
            state_id TEXT PRIMARY KEY,
            state_index INT NOT NULL,
            content JSONB NOT NULL
        );
    """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_trail_id ON trail(id);")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_trail_state_index ON trail(state_index);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_trail_content ON trail USING GIN(content);"
    )

    conn.commit()
    cur.close()
    conn.close()


def load_data(path_to_traces: str):
    tasks = []
    for file in os.listdir(path_to_traces):
        with open(os.path.join(path_to_traces, file), "r") as f:
            data = json.load(f)
            tasks.append({"trace_id": data["trace_id"], "history": data["history"]})

    df = pd.json_normalize(tasks)

    return df


def transform(df):
    rows = []

    for _, task in df.iterrows():
        trace_id = task["trace_id"]
        history = task["history"]

        for i, state in enumerate(history, 1):
            state_id = f"{trace_id}_{i}"

            content = json.dumps(state)

            rows.append((trace_id, state_id, i, content))

    return rows


def insert_rows(rows):
    if not rows:
        print("no rows")
        return

    conn = get_conn(DB_NAME)
    cur = conn.cursor()

    q = """
        INSERT INTO trail (id, state_id, state_index, content)
        VALUES %s
        ON CONFLICT (state_id) DO NOTHING;
    """

    execute_values(cur, q, rows, page_size=1000)

    conn.commit()
    cur.close()
    conn.close()


def load_and_insert(path_to_traces: str):
    print(f"loading {path_to_traces} ...")
    df = load_data(path_to_traces)

    print("transform...")
    rows = transform(df)

    print(f"inserting {len(rows)} rows")
    insert_rows(rows)


def main():
    create_database()
    create_table()

    path_to_traces = "path/to/your/directory"

    load_and_insert(path_to_traces)

    print("done")

if __name__ == "__main__":
    drop_table()
    main()
