import psycopg2
from psycopg2.extras import execute_values
import json
from datasets import load_dataset

# db config
DB_NAME = "maseval"
DB_USER = "postgres"
DB_PASSWORD = ""  # set to None if no password
DB_HOST = "localhost"
DB_PORT = 5432


def drop_table():
    conn = get_conn(DB_NAME)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS aegis;")
    conn.commit()

    cur.close()
    conn.close()

    print("aegis dropped")


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
        CREATE TABLE IF NOT EXISTS aegis (
            id TEXT NOT NULL,
            state_id TEXT PRIMARY KEY,
            state_index INT NOT NULL,
            content JSONB NOT NULL
        );
    """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_aegis_id ON aegis(id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_aegis_state_index ON aegis(state_index);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_aegis_content ON aegis USING GIN(content);")

    conn.commit()
    cur.close()
    conn.close()


def transform(ds):
    rows = []

    for r in ds:
        trace_id = r["id"]
        history = r["input"]["conversation_history"]

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
        INSERT INTO aegis (id, state_id, state_index, content)
        VALUES %s
        ON CONFLICT (state_id) DO NOTHING;
    """

    execute_values(cur, q, rows, page_size=1000)

    conn.commit()
    cur.close()
    conn.close()


def load_and_insert(split: str):
    print(f"loading AEGIS split {split} ...")
    ds = load_dataset("Fancylalala/AEGIS", split=split)

    print("transform...")
    rows = transform(ds)

    print(f"inserting {len(rows)} rows")
    insert_rows(rows)


def main():
    create_database()
    create_table()

    load_and_insert("test")
    # Additional splits can be loaded here if needed

    print("done")


if __name__ == "__main__":
    drop_table()
    main()
