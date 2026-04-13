import psycopg2
from psycopg2.extras import execute_values

from automas.db.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER


def get_conn(db: str = DB_NAME) -> psycopg2.extensions.connection:
    try:
        return psycopg2.connect(
            dbname=db, user=DB_USER, password=DB_PASSWORD or None, host=DB_HOST, port=DB_PORT
        )
    except psycopg2.OperationalError as e:
        if "does not exist" in str(e):
            return psycopg2.connect(
                dbname="postgres",
                user=DB_USER,
                password=DB_PASSWORD or None,
                host=DB_HOST,
                port=DB_PORT,
            )
        raise


def create_database() -> None:
    conn = psycopg2.connect(
        dbname="postgres", user=DB_USER, password=DB_PASSWORD or None, host=DB_HOST, port=DB_PORT
    )
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


def drop_table(table_name: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {table_name};")
    conn.commit()
    cur.close()
    conn.close()
    print(f"{table_name} dropped")


def create_table(table_name: str) -> None:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT NOT NULL,
            state_id TEXT PRIMARY KEY,
            state_index INT NOT NULL,
            content JSONB NOT NULL
        );
        """
    )

    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_id ON {table_name}(id);")
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_state_index ON {table_name}(state_index);"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_content ON {table_name} USING GIN(content);"
    )

    conn.commit()
    cur.close()
    conn.close()


def insert_rows(table_name: str, rows: list) -> None:
    if not rows:
        print("No rows to insert")
        return

    conn = get_conn()
    cur = conn.cursor()

    q = f"""
        INSERT INTO {table_name} (id, state_id, state_index, content)
        VALUES %s
        ON CONFLICT (state_id) DO NOTHING;
    """

    execute_values(cur, q, rows, page_size=1000)

    conn.commit()
    cur.close()
    conn.close()

    print(f"Inserted {len(rows)} rows into {table_name}")
