import json
import os

from .base import create_database, create_table, drop_table, insert_rows

TABLE_NAME = "trail"


def load_data(path_to_traces: str):
    tasks = []
    for file in os.listdir(path_to_traces):
        with open(os.path.join(path_to_traces, file), "r") as f:
            data = json.load(f)
            tasks.append({"trace_id": data["trace_id"], "history": data["history"]})
    return tasks


def transform(tasks):
    rows = []
    for task in tasks:
        trace_id = task["trace_id"]
        history = task["history"]
        for i, state in enumerate(history, 1):
            state_id = f"{trace_id}_{i}"
            rows.append((trace_id, state_id, i, json.dumps(state)))
    return rows


def load_and_insert(path_to_traces: str):
    print(f"Loading {path_to_traces} ...")
    tasks = load_data(path_to_traces)

    print("Transforming...")
    rows = transform(tasks)

    print(f"Inserting {len(rows)} rows")
    insert_rows(TABLE_NAME, rows)


def main():
    create_database()
    create_table(TABLE_NAME)

    path_to_traces = "path/to/your/directory"
    load_and_insert(path_to_traces)

    print("Done")


if __name__ == "__main__":
    drop_table(TABLE_NAME)
    main()
