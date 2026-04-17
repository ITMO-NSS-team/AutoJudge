import json

import pandas as pd

from .base import create_database, create_table, drop_table, insert_rows

TABLE_NAME = "who_when"


def load_data(fname: str):
    path = f"hf://datasets/Kevin355/Who_and_When/{fname}"
    return pd.read_parquet(path)


def transform(df):
    rows = []
    for _, r in df.iterrows():
        trace_id = r["question_ID"]
        history = r["history"]
        for i, state in enumerate(history, 1):
            state_id = f"{trace_id}_{i}"
            rows.append((trace_id, state_id, i, json.dumps(state)))
    return rows


def load_and_insert(fname: str):
    print(f"Loading {fname} ...")
    df = load_data(fname)

    print("Transforming...")
    rows = transform(df)

    print(f"Inserting {len(rows)} rows")
    insert_rows(TABLE_NAME, rows)


def main():
    create_database()
    create_table(TABLE_NAME)

    load_and_insert("Hand-Crafted.parquet")
    load_and_insert("Algorithm-Generated.parquet")

    print("Done")


if __name__ == "__main__":
    drop_table(TABLE_NAME)
    main()
