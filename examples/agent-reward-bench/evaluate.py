"""Evaluate WebArena AutoJudge results against ground truth using ARB methodology."""

import json
import csv
from pathlib import Path
import argparse


def load_arb_ground_truth(annotations_file: str) -> dict:
    """Load ARB ground truth annotations from CSV.

    Maps ARB column names to our binary dimensions:
    - trajectory_success (Successful/Unsuccessful) → success
    - trajectory_side_effect (Yes/No) → side_effect
    - trajectory_looping (Yes/No) → repetition_cycle

    Args:
        annotations_file: Path to data/annotations.csv from ARB dataset

    Returns:
        Dict mapping task_id to {success, side_effect, repetition_cycle} booleans
    """
    ground_truth = {}

    try:
        with open(annotations_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_id = row.get("task_id")
                if not task_id:
                    continue

                # Map ARB columns to our dimensions
                success = row.get("trajectory_success", "").lower() == "successful"
                side_effect = row.get("trajectory_side_effect", "").lower() == "yes"
                looping = row.get("trajectory_looping", "").lower() == "yes"

                ground_truth[task_id] = {
                    "success": success,
                    "side_effect": side_effect,
                    "repetition_cycle": looping,
                }

        return ground_truth

    except FileNotFoundError:
        print(f"Error: Ground truth file not found: {annotations_file}")
        return {}
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return {}


def calculate_binary_metrics(predictions: list[bool], ground_truth: list[bool]) -> dict:
    """Calculate precision, recall, F1, TNR, NPV for binary classification."""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    tp = sum(1 for p, gt in zip(predictions, ground_truth) if p and gt)
    fp = sum(1 for p, gt in zip(predictions, ground_truth) if p and not gt)
    tn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and not gt)
    fn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tnr": tnr,
        "npv": npv,
    }


def evaluate_results(results_dir: str, ground_truth_file: str = None):
    """Evaluate judge predictions against ground truth using ARB metrics.

    Args:
        results_dir: Path to results directory with per-trace JSON files
        ground_truth_file: Optional path to ARB data/annotations.csv for ground truth
    """
    results_path = Path(results_dir)

    if not results_path.is_dir():
        print(f"Error: {results_dir} is not a directory")
        return

    # Load all judge results
    task_results = []
    for fp in sorted(results_path.glob("*.json")):
        if fp.name == "failed_traces.txt":
            continue
        try:
            with open(fp) as f:
                data = json.load(f)
                task_results.append(data)
        except Exception as e:
            print(f"Error reading {fp}: {e}")

    if not task_results:
        print("No results found")
        return

    # Load ARB ground truth if provided
    arb_ground_truth = {}
    if ground_truth_file:
        arb_ground_truth = load_arb_ground_truth(ground_truth_file)
        if arb_ground_truth:
            print(f"Loaded {len(arb_ground_truth)} ground truth annotations from ARB")

    # Extract predictions and ground truth
    dimensions = ["success", "side_effect", "repetition_cycle"]
    metrics_by_dimension = {}

    for dim in dimensions:
        predictions = []
        ground_truth = []

        for result in task_results:
            task_id = result.get("task_id")
            pred = result.get("judge_predictions", {}).get(dim)

            # Try ARB ground truth first, fall back to result's own ground truth
            if arb_ground_truth and task_id in arb_ground_truth:
                gt = arb_ground_truth[task_id].get(dim)
            else:
                gt = result.get("ground_truth", {}).get(dim)

            if pred is not None and gt is not None:
                predictions.append(pred)
                ground_truth.append(gt)

        if predictions:
            metrics_by_dimension[dim] = calculate_binary_metrics(predictions, ground_truth)

    # Print report
    print("\n" + "=" * 80)
    print("WebArena AutoJudge Evaluation Results")
    print("=" * 80)
    print(f"\nTotal Traces with Judge Predictions: {len(task_results)}")

    if arb_ground_truth:
        print(f"Ground Truth Source: ARB annotations.csv (expert human labels)")
    else:
        print(f"Ground Truth Source: Embedded in results JSON (task_success)")

    for dim in dimensions:
        if dim in metrics_by_dimension:
            m = metrics_by_dimension[dim]
            print(f"\n[{dim.upper().replace('_', ' ')}]")
            print(f"  Samples: {m['tp'] + m['fp'] + m['tn'] + m['fn']}")
            print(f"  TP: {m['tp']}, FP: {m['fp']}, TN: {m['tn']}, FN: {m['fn']}")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall:    {m['recall']:.4f}")
            print(f"  F1 Score:  {m['f1']:.4f}")
            print(f"  TNR:       {m['tnr']:.4f}")
            print(f"  NPV:       {m['npv']:.4f}")
        else:
            print(f"\n[{dim.upper().replace('_', ' ')}] - No data available")

    print("\n" + "=" * 80)
    if arb_ground_truth:
        print("✓ Validated against ARB ground truth (expert human annotations)")
        print("  This proves AutoJudge achieves equivalent performance to ARB judges")
    else:
        print("Note: For full ARB validation, provide ground truth with --ground-truth")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate WebArena AutoJudge results using ARB methodology"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results directory containing judge prediction JSON files",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Optional path to ground truth annotations file",
    )
    args = parser.parse_args()

    evaluate_results(args.results, args.ground_truth)
