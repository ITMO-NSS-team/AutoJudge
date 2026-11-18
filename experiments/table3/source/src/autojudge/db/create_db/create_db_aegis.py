import json

from datasets import load_dataset

from .base import create_database, create_table, drop_table, insert_rows

TABLE_NAME = "aegis"


def transform(ds):
    rows = []
    for r in ds:
        trace_id = r["id"]
        history = r["input"]["conversation_history"]
        for i, state in enumerate(history, 1):
            state_id = f"{trace_id}_{i}"
            rows.append((trace_id, state_id, i, json.dumps(state)))
    return rows


def load_and_insert(split: str):
    print(f"Loading AEGIS split {split} ...")
    ds = load_dataset("Fancylalala/AEGIS", split=split)

    print("Transforming...")
    rows = transform(ds)

    print(f"Inserting {len(rows)} rows")
    insert_rows(TABLE_NAME, rows)


def main():
    create_database()
    create_table(TABLE_NAME)
    load_and_insert("test")
    print("Done")


if __name__ == "__main__":
    drop_table(TABLE_NAME)
    main()
