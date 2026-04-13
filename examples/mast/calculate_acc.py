import json
from collections import defaultdict
from pathlib import Path
import csv
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Optional
import re


def parse_failure_mode_key(fm_text: str) -> Optional[str]:
    """Extract failure mode key (e.g. '1.1', '2.5') from text"""
    match = re.match(r"^(\d+\.\d+)", fm_text.strip())
    return match.group(1) if match else None


def load_single_json(filepath: str) -> Dict:
    """Load single JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics_from_directory(directory: str) -> tuple[Dict, int]:
    """Compute metrics for each failure mode type from directory with separate JSONs"""
    directory_path = Path(directory)
    json_files = list(directory_path.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in directory {directory}")

    print(f"Found {len(json_files)} JSON files")

    # collect all unique failure modes from evaluation_results
    eval_fms = set()
    for file_path in json_files:
        try:
            data = load_single_json(str(file_path))
            scores = data.get("evaluation_results", {}).get("scores", [])
            for score in scores:
                key = parse_failure_mode_key(score.get("failure mode", ""))
                if key:
                    eval_fms.add(key)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    print(f"Unique failure modes in evaluation_results: {len(eval_fms)}")

    # dictionary to store results
    results = defaultdict(
        lambda: {"preds": [], "gts": [], "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    )

    processed_files = 0
    for file_path in json_files:
        try:
            data = load_single_json(str(file_path))
            processed_files += 1

            # evaluation_results -> predictions
            eval_scores = data.get("evaluation_results", {}).get("scores", [])
            eval_dict = {}
            for score in eval_scores:
                key = parse_failure_mode_key(score.get("failure mode", ""))
                if key:
                    eval_dict[key] = score.get("result", False)

            # metadata.annotations -> ground truth (majority vote)
            annotations = data.get("metadata", {}).get("annotations", [])
            annot_majority = {}
            for ann in annotations:
                key = parse_failure_mode_key(ann.get("failure mode", ""))
                if key:
                    votes = sum(
                        [
                            ann.get("annotator_1", False),
                            ann.get("annotator_2", False),
                            ann.get("annotator_3", False),
                        ]
                    )
                    majority = votes >= 2
                    annot_majority[key] = majority

            # compute metrics for each fm
            for fm_key in eval_fms:
                pred = eval_dict.get(fm_key, False)
                gt = annot_majority.get(fm_key)
                if gt is not None:
                    results[fm_key]["preds"].append(pred)
                    results[fm_key]["gts"].append(gt)

                    if gt and pred:  # TP
                        results[fm_key]["tp"] += 1
                    elif not gt and not pred:  # TN
                        results[fm_key]["tn"] += 1
                    elif gt and not pred:  # FN
                        results[fm_key]["fn"] += 1
                    elif not gt and pred:  # FP
                        results[fm_key]["fp"] += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    print(f"Processed files: {processed_files}")

    # calculate metrics
    metrics = {}
    for fm_key, stats in results.items():
        if stats["gts"]:  # has ground truth
            acc = accuracy_score(stats["gts"], stats["preds"])
            f1 = f1_score(stats["gts"], stats["preds"], zero_division=0)

            metrics[fm_key] = {
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
                "tp": stats["tp"],
                "fp": stats["fp"],
                "tn": stats["tn"],
                "fn": stats["fn"],
                "total_count": len(stats["gts"]),
            }

    return metrics, processed_files


def save_to_csv(metrics: Dict, csv_path: Path):
    """Save metrics to CSV file"""
    fieldnames = ["FM", "Accuracy", "F1", "TP", "FP", "TN", "FN"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # sort by failure mode number
        sorted_metrics = dict(
            sorted(metrics.items(), key=lambda x: tuple(map(float, x[0].split("."))))
        )

        for fm_key, m in sorted_metrics.items():
            writer.writerow(
                {
                    "FM": fm_key,
                    "Accuracy": m["accuracy"],
                    "F1": m["f1_score"],
                    "TP": m["tp"],
                    "FP": m["fp"],
                    "TN": m["tn"],
                    "FN": m["fn"],
                }
            )


def print_results_table(metrics: Dict, total_files: int):
    """Print results table: FM, Acc, F1, TP, FP, TN, FN"""
    if not metrics:
        print("No metrics data available!")
        return

    print("\n" + "=" * 80)
    print("FAILURE MODES METRICS")
    print("=" * 80)
    print(f"{'FM':<4} {'Acc':<7} {'F1':<6} {'TP':<3} {'FP':<3} {'TN':<4} {'FN':<3}")
    print("-" * 80)

    # sort by failure mode number
    sorted_metrics = dict(
        sorted(metrics.items(), key=lambda x: tuple(map(float, x[0].split("."))))
    )

    for fm_key, m in sorted_metrics.items():
        print(
            f"{fm_key:<4} {m['accuracy']:<6.3f} {m['f1_score']:<5.3f} "
            f"{m['tp']:<3} {m['fp']:<3} {m['tn']:<4} {m['fn']:<3}"
        )

    print("-" * 80)


def main(directory):
    try:
        metrics, total_files = compute_metrics_from_directory(directory)
        print_results_table(metrics, total_files)

        json_path = Path(directory) / "failure_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        csv_path = Path(directory) / "failure_metrics.csv"
        save_to_csv(metrics, csv_path)

        print(f"\nJSON saved: {json_path}")
        print(f"CSV saved:  {csv_path}")
        print(f"Total failure modes with metrics: {len(metrics)}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main(
        "/Users/alina/Desktop/ITMO/AutoJudge/examples/mast/results/mast_human_dataset_eval"
    )
