"""Calculate metrics according to ARB using WebArena AutoJudge results."""
import json
import csv
from pathlib import Path
import argparse

def load_arb_ground_truth(annotations_file: str) -> dict:
    ground_truth = {}
    try:
        with open(annotations_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_id = row.get("task_id")
                model_name = row.get("model_name")
                if not task_id or not model_name:
                    continue

                unique_task_id = f"{model_name}_{task_id}"

                success = row.get("trajectory_success", "").lower() == "successful"
                side_effect = row.get("trajectory_side_effect", "").lower() == "yes"
                looping = row.get("trajectory_looping", "").lower() == "yes"

                ground_truth[unique_task_id] = {
                    "success": success,
                    "side_effect": side_effect,
                    "repetition_cycle": looping,
                }
        return ground_truth
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return {}

def calculate_binary_metrics(predictions: list[bool], ground_truth: list[bool]) -> dict:
    """Calculate ARB metrics. For reference, see:
    https://github.com/McGill-NLP/agent-reward-bench/blob/main/scripts/score_judgments.py
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    tp = sum(1 for p, gt in zip(predictions, ground_truth) if p and gt)
    fp = sum(1 for p, gt in zip(predictions, ground_truth) if p and not gt)
    tn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and not gt)
    fn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0 
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "tnr": tnr, "npv": npv, "samples": len(predictions)
    }

def evaluate_results(results_dir: str, ground_truth_file: str = None):
    results_path = Path(results_dir)
    if not results_path.is_dir():
        print(f"Error: {results_dir} is not a directory")
        return

    task_results = []
    for fp in sorted(results_path.glob("*.json")):
        if fp.name == "failed_traces.txt":
            continue
        try:
            with open(fp) as f:
                task_results.append(json.load(f))
        except Exception as e:
            print(f"Error reading {fp}: {e}")

    if not task_results:
        print("No results found")
        return

    arb_ground_truth = {}
    if ground_truth_file:
        arb_ground_truth = load_arb_ground_truth(ground_truth_file)

    dimensions = ["success", "side_effect", "repetition_cycle"]
    metrics_by_dimension = {}

    for dim in dimensions:
        predictions = []
        ground_truth = []

        for result in task_results:
            task_id = result.get("task_id")
            pred_dim = "trajectory_looping" if dim == "repetition_cycle" else f"trajectory_{dim}"
            pred = result.get("judge_predictions", {}).get(pred_dim)

            # We use standard ARB ids (e.g. webarena.185) natively now
            gt = None
            if arb_ground_truth and task_id in arb_ground_truth:
                gt = arb_ground_truth[task_id].get(dim)
            else:
                gt = result.get("ground_truth", {}).get(dim)

            if pred is not None and gt is not None:
                predictions.append(pred)
                ground_truth.append(gt)

        if predictions:
            metrics_by_dimension[dim] = calculate_binary_metrics(predictions, ground_truth)

    print("\n" + "=" * 80)
    print("WebArena AutoJudge Evaluation Results")
    print("=" * 80)
    print(f"\nTotal Traces with Judge Predictions: {len(task_results)}")

    print(f"Ground Truth Source: {'ARB annotations.csv' if arb_ground_truth else 'Embedded JSON info'}")

    for dim in dimensions:
        print(f"\n[{dim.upper().replace('_', ' ')}]")
        if dim in metrics_by_dimension:
            m = metrics_by_dimension[dim]
            print(f"  Samples:   {m['samples']}")
            print(f"  Confusion: TP: {m['tp']}, FP: {m['fp']}, TN: {m['tn']}, FN: {m['fn']}")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall:    {m['recall']:.4f}")
            print(f"  F1 Score:  {m['f1']:.4f}")
            print(f"  TNR:       {m['tnr']:.4f}")
            print(f"  NPV:       {m['npv']:.4f}")
        else:
            print("  No data available")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=str(Path(__file__).resolve().parent / "results" / "webarena_results"), help="Path to results directory")
    parser.add_argument("--ground-truth", type=str, default=str(Path(__file__).resolve().parent / "data" / "annotations.csv"), help="Path to annotations.csv")
    args = parser.parse_args()
    evaluate_results(args.results, args.ground_truth)
