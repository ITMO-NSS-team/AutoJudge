import os
from pathlib import Path
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

    cur.execute("DROP TABLE IF EXISTS webarena;")
    conn.commit()

    cur.close()
    conn.close()

    print("webarena dropped")


def get_conn(db):
    try:
        return psycopg2.connect(
            dbname=db, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
        )
    except psycopg2.OperationalError as e:
        if "does not exist" in str(e):
            return psycopg2.connect(
                dbname="postgres", user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
            )
        raise


def create_database():
    conn = psycopg2.connect(dbname="postgres", user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
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
        CREATE TABLE IF NOT EXISTS webarena (
            id TEXT NOT NULL,
            state_id TEXT PRIMARY KEY,
            state_index INT NOT NULL,
            content JSONB NOT NULL
        );
    """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_webarena_id ON webarena(id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_webarena_state_index ON webarena(state_index);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_webarena_content ON webarena USING GIN(content);")

    conn.commit()
    cur.close()
    conn.close()


def transform(data_dir: Path):
    rows = []
    
    trace_files = list(data_dir.rglob("*.json"))
    for trace_file in trace_files:
        with open(trace_file, "r", encoding="utf-8") as f:
            trace_data = json.load(f)
            
        trace_id = trace_file.stem
        steps = trace_data.get("steps", [])

        for i, step in enumerate(steps, 1):
            state_id = f"{trace_id}_{i}"
            rows.append((str(trace_id), str(state_id), i, json.dumps(step)))

    return rows


def insert_rows(rows):
    if not rows:
        print("no rows")
        return

    conn = get_conn(DB_NAME)
    cur = conn.cursor()

    query = """
        INSERT INTO webarena (id, state_id, state_index, content)
        VALUES %s
        ON CONFLICT (state_id) DO NOTHING;
    """

    execute_values(cur, query, rows, template=None, page_size=100)

    conn.commit()
    cur.close()
    conn.close()

    print(f"inserted {len(rows)} nodes")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="examples/agent-reward-bench/data/cleaned", help="Path to cleaned directory containing trace files")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    data_dir_path = project_root / args.data_dir

    create_database()
    drop_table()
    create_table()

    if data_dir_path.exists():
        rows = transform(data_dir_path)
        insert_rows(rows)
    else:
        print(f"Data directory {data_dir_path} does not exist.")