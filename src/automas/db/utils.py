import psycopg2
import json

DB_NAME = "maseval"
DB_USER = "postgres"
DB_PASSWORD = ""  # set to None if no password
DB_HOST = "localhost"
DB_PORT = 5432

def get_content_by_state(state_id, table_name="our_mas"):
    """
    Fetches the 'content' field from the table for a given state_id.
    Returns a Python dict or None if not found.
    """
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
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
    # our_mas state_id: "0d1d4b2d2c1dd706b347aaa05d29d503_1"
    # who_when state_id: "5f982798-16b9-4051-ab57-cfc7ebdb2a91_1"
    test_state_id = "0d1d4b2d2c1dd706b347aaa05d29d503_1" 
    content = get_content_by_state(
        state_id=test_state_id,
        table_name='our_mas') # or who_when
    if content:
        print(json.dumps(content, indent=2, ensure_ascii=False))