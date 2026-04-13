import json
import os
from pathlib import Path

from .base import create_database, create_table, drop_table, insert_rows

TABLE_NAME = "agent_error"


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
                tasks.append({"trace_id": trace_id, "messages": messages})
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return tasks


def transform(tasks):
    rows = []
    for task in tasks:
        trace_id = task["trace_id"]
        messages = task["messages"]
        if not isinstance(messages, list):
            continue
        for i, state in enumerate(messages, start=1):
            state_id = f"{trace_id}_{i}"
            content = json.dumps(state, ensure_ascii=False)
            rows.append((trace_id, state_id, i, content))
    return rows


def load_and_insert(path_to_traces: str):
    print(f"Loading from {path_to_traces} ...")
    tasks = load_data(path_to_traces)

    print("Transforming...")
    rows = transform(tasks)

    print(f"Inserting {len(rows)} rows")
    insert_rows(TABLE_NAME, rows)


def main():
    create_database()
    create_table(TABLE_NAME)

    path_to_traces = "path/to/your/traces"
    load_and_insert(path_to_traces)

    print("Done")


if __name__ == "__main__":
    drop_table(TABLE_NAME)
    main()
