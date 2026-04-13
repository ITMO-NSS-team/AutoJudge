import json

import psycopg2

from .config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER


def get_content_by_state(state_id, table_name="our_mas"):
    """
    Fetches the 'content' field from the table for a given state_id.
    Returns a Python dict or None if not found.
    """
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD or None,
            host=DB_HOST,
            port=DB_PORT,
        )
        cur = conn.cursor()

        query = f"SELECT content FROM {table_name} WHERE state_id = %s;"
        cur.execute(query, (state_id,))
        result = cur.fetchone()

        cur.close()
        conn.close()

        if result:
            return str(result)
        else:
            print(f"State_id not found: {state_id}")
            return None

    except Exception as e:
        print(f"Error fetching content: {e}")
        return None


if __name__ == "__main__":
    test_state_id = "0adc4f3b99d9564d32811e913cc9d248_1"
    content = get_content_by_state(state_id=test_state_id, table_name="our_mas")
    if content:
        print(json.dumps(content, indent=2, ensure_ascii=False))
