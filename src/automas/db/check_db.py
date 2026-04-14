import json

import psycopg2

from .config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER


def check_table(table_name: str = "our_mas", limit: int = 5):
    """Checks the content of the specified table and prints the first few rows."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD or None,
            host=DB_HOST,
            port=DB_PORT,
        )
        cur = conn.cursor()

        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        total = cur.fetchone()[0]
        print(f"Found rows {table_name}: {total}")

        cur.execute(
            f"SELECT id, state_id, state_index, content FROM {table_name} LIMIT %s;",
            (limit,),
        )
        rows = cur.fetchall()

        print(f"\nFirst {limit} rows:")
        for row in rows:
            id_val, state_id, state_index, content = row
            print(f"\nid: {id_val}")
            print(f"state_id: {state_id}")
            print(f"state_index: {state_index}")
            print(
                f"content: {json.dumps(content, indent=2, ensure_ascii=False)[:200]}..."
            )

        cur.close()
        conn.close()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    check_table(table_name="agent_error")  # or who_when or trail
