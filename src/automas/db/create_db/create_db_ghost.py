import json

from maseval import get_langfuse_download_client
from maseval.parsers.langfuse_parser_v3 import parse_langfuse_task

from .base import create_database, create_table, drop_table, insert_rows

TABLE_NAME = "our_mas"


def load_data(name: str):
    lf = get_langfuse_download_client()

    all_traces = []
    for page in range(1, 5):
        traces = lf.api.trace.list(name=name, limit=50, page=page)
        all_traces.extend(traces.data)

    tasks = []
    for item in all_traces:
        try:
            task = parse_langfuse_task(lf.api.trace.get(item.id))
            if task.agent_states:
                tasks.append((item.id, task))
        except Exception:
            print(f"Task {item.id} has errors")
            continue

    return tasks


def transform(tasks):
    rows = []
    for task_id, task in tasks:
        for i, state in enumerate(task.agent_states, 1):
            state_id = f"{task_id}_{i}"
            content = json.dumps(state.model_dump(mode="json"))
            rows.append((task_id, state_id, i, content))
    return rows


def transform_big_mas(tasks):
    rows = []
    for task_id, task in tasks:
        for i, state in enumerate(task.agent_states, 1):
            state_id = f"{task_id}_{i}"
            content = {
                "type": state.type,
                "content": state.content,
                "metadata": state.metadata,
                "state_id": state.state_id,
                "timestamp": state.timestamp,
            }
            content = json.dumps(content, default=str).replace("\\u0000", "")
            rows.append((task_id, state_id, i, content))
    return rows


def load_and_insert(name: str):
    print(f"Loading {name} ...")
    tasks = load_data(name)

    print("Transforming...")
    if name == "gaia_task_db0c3ed0-a4af-4442-bb6f-884d6da055cb":
        rows = transform_big_mas(tasks)
    else:
        rows = transform(tasks)

    print(f"Inserting {len(rows)} rows")
    insert_rows(TABLE_NAME, rows)


def main():
    create_database()
    create_table(TABLE_NAME)

    load_and_insert("gaia_task_07aac7b1-ffc3-4787-8e4c-7fb522156097")  # small mas
    load_and_insert("gaia_task_db0c3ed0-a4af-4442-bb6f-884d6da055cb")  # big mas

    print("Done")


if __name__ == "__main__":
    drop_table(TABLE_NAME)
    main()
