"""
Calculate binary classification metrics for summarizer-only result files.

Expected result JSON schema per task:
{
  "summarizer_score": {
    "metric_name": "summarizer_score",
    "scores": [
      {
        "item_id": "overall_score",
        "score": "ideal|fair|poor",
        "justification": "..."
      }
    ]
  }
}
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")

SCORE_MAPPING = {"ideal": 1, "fair": 0, "poor": 0}


def calculate_f1_score(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def normalize_gt_label(value) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        if value in (0, 1):
            return int(value)
        return None

    text = str(value).strip().lower()
    if text in {"1", "true", "ideal", "yes"}:
        return 1
    if text in {"0", "false", "poor", "fair", "no"}:
        return 0

    return None


def parse_summarizer_results(
    directory_path: str, metric_name: str = "summarizer_score"
) -> pd.DataFrame:
    directory = Path(directory_path)
    json_files = [p for p in directory.glob("*.json") if p.name != "failed_traces.txt"]

    rows: list[dict] = []
    missing_score_files: list[str] = []

    print(f"Found {len(json_files)} JSON files to process")

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = json.load(file)

            task_id = file_path.stem.strip().lower()
            score_items = content.get(metric_name, {}).get("scores", [])

            raw_score = ""
            justification = ""
            if score_items:
                raw_score = str(score_items[0].get("score", "")).strip().lower()
                justification = str(score_items[0].get("justification", ""))

            if raw_score == "":
                missing_score_files.append(file_path.name)
                continue

            if raw_score not in SCORE_MAPPING:
                missing_score_files.append(file_path.name)
                continue

            rows.append(
                {
                    "task_id": task_id,
                    "raw_score": raw_score,
                    "judge_score": SCORE_MAPPING[raw_score],
                    "justification": justification,
                }
            )

        except Exception as exc:
            print(f"Error processing file {file_path}: {exc}")

    if missing_score_files:
        print(
            f"Files with missing/unknown summarizer score: {len(missing_score_files)}"
        )

    return pd.DataFrame(rows)


def evaluate_against_gt(
    df_results: pd.DataFrame, human_gt_path: str, encoding: str = "ISO-8859-1"
):
    gt_df = pd.read_csv(human_gt_path, encoding=encoding)

    required_columns = {"trace_id", "ht"}
    missing_columns = required_columns - set(gt_df.columns)
    if missing_columns:
        raise ValueError(
            "HUMAN_GT_PATH CSV must contain columns: trace_id, ht. "
            f"Missing: {sorted(missing_columns)}"
        )

    gt_df = gt_df.copy()
    gt_df = gt_df[gt_df["trace_id"].notna()].copy()
    gt_df["task_id"] = gt_df["trace_id"].astype(str).str.strip().str.lower()
    gt_df["gt_label"] = gt_df["ht"].apply(normalize_gt_label)
    gt_df = gt_df[["task_id", "gt_label"]]

    merged = df_results.merge(gt_df, on="task_id", how="left")
    merged = merged[merged["gt_label"].notna()].copy()
    merged["gt_label"] = merged["gt_label"].astype(int)
    merged["is_correct"] = merged["judge_score"] == merged["gt_label"]

    tp = int(((merged["judge_score"] == 1) & (merged["gt_label"] == 1)).sum())
    fp = int(((merged["judge_score"] == 1) & (merged["gt_label"] == 0)).sum())
    fn = int(((merged["judge_score"] == 0) & (merged["gt_label"] == 1)).sum())
    tn = int(((merged["judge_score"] == 0) & (merged["gt_label"] == 0)).sum())

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = calculate_f1_score(tp, fp, fn)

    return merged, {
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


if __name__ == "__main__":
    result_dir = "results/big_mas_no_sum"
    output_csv = "metrics_summarizer_only.csv"
    default_gt_path = "gaia_task_db0c3ed0.csv"

    human_gt_path = os.environ.get("HUMAN_GT_PATH", default_gt_path)
    print(f"Using GT file: {human_gt_path}")

    print("Parsing summarizer-only files...")
    result_df = parse_summarizer_results(result_dir, metric_name="summarizer_score")

    if result_df.empty:
        raise ValueError("No valid summarizer scores found")

    eval_df, stats = evaluate_against_gt(result_df, human_gt_path)

    if stats["total"] == 0:
        result_ids = set(result_df["task_id"].astype(str))
        gt_preview_df = pd.read_csv(human_gt_path, encoding="ISO-8859-1")
        gt_ids = set(
            gt_preview_df["trace_id"].dropna().astype(str).str.strip().str.lower()
        )
        overlap = result_ids & gt_ids
        raise ValueError(
            "No overlapping task IDs between results and HUMAN_GT_PATH. "
            f"results={len(result_ids)}, gt={len(gt_ids)}, overlap={len(overlap)}. "
            f"Sample result IDs: {list(sorted(result_ids))[:3]}; "
            f"sample GT IDs: {list(sorted(gt_ids))[:3]}"
        )

    eval_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 50)
    print("METRICS FOR summarizer_score")
    print("=" * 50)
    print(f"Rows evaluated: {stats['total']}")
    print(f"F1: {stats['f1']:.4f}")
    print(f"Accuracy: {stats['accuracy']:.4f}")
    print(f"Precision: {stats['precision']:.4f}")
    print(f"Recall: {stats['recall']:.4f}")
    print(f"TP: {stats['tp']}")
    print(f"FP: {stats['fp']}")
    print(f"FN: {stats['fn']}")
    print(f"TN: {stats['tn']}")
    print(f"Saved detailed results to: {output_csv}")
