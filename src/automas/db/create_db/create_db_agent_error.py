import os
import json
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv(".env")


DB_NAME = "maseval"
DB_USER = "postgres"
DB_PASSWORD = ""  # None if no password
DB_HOST = "localhost"
DB_PORT = 5432


def get_conn(db):
    return psycopg2.connect(
        dbname=db,
        user=DB_USER,
        password=DB_PASSWORD if DB_PASSWORD else None,
        host=DB_HOST,
        port=DB_PORT,
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


def drop_table():
    conn = get_conn(DB_NAME)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS agent_error;")
    conn.commit()

    cur.close()
    conn.close()
    print("agent_error dropped")


def create_table():
    conn = get_conn(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_error (
            id TEXT NOT NULL,
            state_id TEXT PRIMARY KEY,
            state_index INT NOT NULL,
            content JSONB NOT NULL
        );
        """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_error_id ON agent_error(id);")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_error_state_index ON agent_error(state_index);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_error_content ON agent_error USING GIN(content);"
    )

    conn.commit()
    cur.close()
    conn.close()


def load_data(path_to_traces: str):
    tasks = []

    for file in os.listdir(path_to_traces):
        if not file.endswith(".json"):
            continue

        filepath = os.path.join(path_to_traces, file)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

                trace_id = Path(file).stem
                messages = data.get("messages", [])

                tasks.append({
                    "trace_id": trace_id,
                    "messages": messages
                })

        except Exception as e:
            print(f"Error reading {file}: {e}")

    return pd.DataFrame(tasks)


def transform(df):
    rows = []

    for _, task in df.iterrows():
        trace_id = task["trace_id"]
        messages = task["messages"]

        if not isinstance(messages, list):
            continue

        for i, state in enumerate(messages, start=1):
            state_id = f"{trace_id}_{i}"
            content = json.dumps(state, ensure_ascii=False)

            rows.append((trace_id, state_id, i, content))

    return rows


def insert_rows(rows):
    if not rows:
        print("No rows to insert")
        return

    conn = get_conn(DB_NAME)
    cur = conn.cursor()

    query = """
        INSERT INTO agent_error (id, state_id, state_index, content)
        VALUES %s
        ON CONFLICT (state_id) DO NOTHING;
    """

    execute_values(cur, query, rows, page_size=1000)

    conn.commit()
    cur.close()
    conn.close()

    print(f"Inserted {len(rows)} rows")


def load_and_insert(path_to_traces: str):
    print(f"Loading from {path_to_traces} ...")

    df = load_data(path_to_traces)

    print("Transforming...")
    rows = transform(df)

    print("Inserting...")
    insert_rows(rows)


def main():
    create_database()
    create_table()

    path_to_traces = "/home/alina/Desktop/AutoJudge/examples/agent_error_bench/AgentErrorBench/Original_Failure_Trajectory/ALFWorld"

    load_and_insert(path_to_traces)

    print("Done")


if __name__ == "__main__":
    drop_table()
    main()