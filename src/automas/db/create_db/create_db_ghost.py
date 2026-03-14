import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import json
from maseval.parsers.langfuse_parser_v3 import parse_langfuse_task
from maseval import get_langfuse_download_client

from dotenv import load_dotenv

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

    cur.execute("DROP TABLE IF EXISTS our_mas;")
    conn.commit()

    cur.close()
    conn.close()

    print("our_mas dropped")


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
        CREATE TABLE IF NOT EXISTS our_mas (
            id TEXT NOT NULL,
            state_id TEXT PRIMARY KEY,
            state_index INT NOT NULL,
            content JSONB NOT NULL
        );
    """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_our_mas_id ON our_mas(id);")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_our_mas_state_index ON our_mas(state_index);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_our_mas_content ON our_mas USING GIN(content);"
    )

    conn.commit()
    cur.close()
    conn.close()


def load_data(name: str):
    lf = get_langfuse_download_client()

    traces_page1 = lf.api.trace.list(name=name, limit=50, page=1)
    traces_page2 = lf.api.trace.list(name=name, limit=50, page=2)
    traces_page3 = lf.api.trace.list(name=name, limit=50, page=3)
    traces_page4 = lf.api.trace.list(name=name, limit=50, page=4)

    all_traces = (
        traces_page1.data + traces_page2.data + traces_page3.data + traces_page4.data
    )

    tasks = []
    for item in all_traces:
        try:
            task = parse_langfuse_task(lf.api.trace.get(item.id))
            if task.agent_states:
                tasks.append((item.id, task))
        except:
            print(f"Task {item.id} has errors")
            continue

    return tasks


def transform(tasks):
    rows = []

    for task_id, task in tasks:
        trace_id = task_id
        history = task.agent_states

        for i, state in enumerate(history, 1):
            state_id = f"{trace_id}_{i}"

            content = json.dumps(state.model_dump(mode="json"))

            rows.append((trace_id, state_id, i, content))

    return rows


def transform_big_mas(tasks):
    rows = []

    for task_id, task in tasks:
        trace_id = task_id
        history = task.agent_states

        for i, state in enumerate(history, 1):
            state_id = f"{trace_id}_{i}"

            content = {
                "type": state.type,
                "content": state.content,
                "metadata": state.metadata,
                "state_id": state.state_id,
                "timestamp": state.timestamp,
            }

            content = json.dumps(content, default=str)
            content = content.replace("\\u0000", "")

            rows.append((trace_id, state_id, i, content))

    return rows


def insert_rows(rows):
    if not rows:
        print("no rows")
        return

    conn = get_conn(DB_NAME)
    cur = conn.cursor()

    q = """
        INSERT INTO our_mas (id, state_id, state_index, content)
        VALUES %s
        ON CONFLICT (state_id) DO NOTHING;
    """

    execute_values(cur, q, rows, page_size=1000)

    conn.commit()
    cur.close()
    conn.close()


def load_and_insert(name: str):
    print(f"loading {name} ...")
    df = load_data(name)

    print("transform...")
    if name == "gaia_task_db0c3ed0-a4af-4442-bb6f-884d6da055cb":
        rows = transform_big_mas(df)
    else:
        rows = transform(df)

    print(f"inserting {len(rows)} rows")
    insert_rows(rows)


def main():
    create_database()
    create_table()

    load_and_insert("gaia_task_07aac7b1-ffc3-4787-8e4c-7fb522156097")  # small mas
    load_and_insert("gaia_task_db0c3ed0-a4af-4442-bb6f-884d6da055cb")  # big mas

    print("done")


if __name__ == "__main__":
    drop_table()
    main()
