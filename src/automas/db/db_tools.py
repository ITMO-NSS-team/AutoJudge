from pydantic import BaseModel
from typing import Optional
from pydantic_ai import Tool
import psycopg2
import json
from pydantic_ai import RunContext, Tool
from typing import Annotated

DB_NAME = "maseval"
DB_USER = "postgres"
DB_PASSWORD = ""  # set to None if no password
DB_HOST = "localhost"
DB_PORT = 5432


class GetContentInput(BaseModel):
    state_id: str
    table_name: Optional[str]


class GetContentOutput(BaseModel):
    content: Optional[dict] = None
    message: Optional[str] = None


async def get_content_function(
    ctx: RunContext[str],
    state_id: Annotated[
        str,
        "REQUIRED: Database state_id (format: 'uuid_1', example: '5f982798-16b9-4051-ab57-cfc7ebdb2a91_1')",
    ],
    table_name: Annotated[
        str, "REQUIRED: Table name (valid_options: 'our_mas', who_when"
    ],
) -> GetContentOutput:
    """
    FETCH ORIGINAL COMPLETE STATE CONTENT FROM DATABASE BY state_id

    Retrieves the 'content' field from PostgreSQL database table for given state_id.

    Args:
        state_id (str): REQUIRED state identifier from MAS trace/execution.
                    Format: "<uuid>_<step_number>" (example: "5f982798-16b9-4051-ab57-cfc7ebdb2a91_1")
        table_name (str): Database table (only 'our_mas' or 'who_when' allowed).

    Returns:
        GetContentOutput: Structured response with:
            - content: dict from database OR None if not found
            - message: "Success" or error description

    Example:
        Input:  state_id="5f982798-16b9-4051-ab57-cfc7ebdb2a91_1"
        Output: {"content": {...}, "message": "Success"}

    Raises:
        ValueError: Invalid table_name
        DatabaseError: Connection/query failures
        JSONDecodeError: Invalid JSON in content field
    """
    VALID_TABLES = {"our_mas", "who_when"}
    fallback_table = (
        (VALID_TABLES - {table_name}).pop() if table_name in VALID_TABLES else None
    )

    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
        )
        cur = conn.cursor()

        tables_to_try = [table_name]
        if fallback_table:
            tables_to_try.append(fallback_table)

        for tbl in tables_to_try:
            cur.execute(f"SELECT content FROM {tbl} WHERE state_id = %s", (state_id,))
            result = cur.fetchone()
            if result:
                content = result[0]
                if not isinstance(content, dict):
                    content = json.loads(content)
                cur.close()
                conn.close()
                found_in = f" (found in '{tbl}')" if tbl != table_name else ""
                return GetContentOutput(content=content, message=f"Success{found_in}")

        cur.close()
        conn.close()
        return GetContentOutput(
            content=None,
            message=f"State_id not found in any table ({', '.join(tables_to_try)})",
        )
    except Exception as e:
        return GetContentOutput(content=None, message=str(e))


get_content_tool = Tool(
    get_content_function,
    name="get_content_tool",
    description="Fetch a row's content from DB",
    takes_ctx=True,
)
