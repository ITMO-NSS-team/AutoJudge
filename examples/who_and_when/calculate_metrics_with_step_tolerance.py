import json
import statistics
from pathlib import Path


def normalize(s: str) -> str:
    return s.lower().replace(" ", "").replace("_", "")


def main(folder_path: str):
    folder = Path(folder_path)

    agent_correct = 0
    agent_total = 0

    # Step accuracy at multiple tolerance levels
    TOLERANCES = [0, 1, 2, 3]
    step_correct = {t: 0 for t in TOLERANCES}
    step_early = 0   # pred = gt - 1 (predicted one step too early)
    step_late  = 0   # pred = gt + 1 (predicted one step too late)
    step_total = 0
    gt_steps:   list[int] = []
    pred_steps: list[int] = []
    diffs:      list[int] = []

    for filepath in folder.rglob("*.json"):  # all .json
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            scores = data["summarizer_score"]["scores"]
            for item in scores:
                gt_agent = normalize(item["gt_agent"])
                agent = normalize(item["score"]["agent"])

                agent_total += 1
                agent_is_correct = gt_agent in agent or agent in gt_agent
                if agent_is_correct:
                    agent_correct += 1

                try:
                    # gt_step is 0-based in the dataset; normalize to 1-based to match judge output
                    gt_step = int(item["gt_step"]) + 1      # for eval steps starting from 1 
                    gt_step = int(item["gt_step"])          # for eval steps starting from 0

                    pred_step = int(item["score"]["step"])
                    step_total += 1
                    diff = pred_step - gt_step          # signed
                    gt_steps.append(gt_step)
                    pred_steps.append(pred_step)
                    diffs.append(diff)
                    for t in TOLERANCES:
                        if abs(diff) <= t:
                            step_correct[t] += 1
                    if diff == -1:
                        step_early += 1
                    elif diff == 1:
                        step_late += 1
                except (ValueError, TypeError):
                    # step could not be parsed as int — skip step metrics for this item
                    pass

        except (KeyError, json.JSONDecodeError) as e:
            print(f"Skeep {filepath}: {type(e).__name__}: {e}")
            continue

    acc_agent = agent_correct / agent_total if agent_total > 0 else 0.0
    print(f"Accuracy Agent:          {acc_agent:.4f} ({agent_correct}/{agent_total})")
    print()
    for t in TOLERANCES:
        acc = step_correct[t] / step_total if step_total > 0 else 0.0
        label = f"Accuracy Step (±{t}):" if t > 0 else "Accuracy Step (exact):"
        print(f"  {label:<26} {acc:.4f} ({step_correct[t]}/{step_total})")
    print()
    n = step_total if step_total > 0 else 1
    print(f"  Step off by -1 (early): {step_early / n:.4f} ({step_early}/{step_total})  [pred = gt - 1]")
    print(f"  Step off by +1 (late):  {step_late  / n:.4f} ({step_late}/{step_total})  [pred = gt + 1]")

    if step_total > 0:
        print()
        print(f"  GT step   — mean: {statistics.mean(gt_steps):.2f}, median: {statistics.median(gt_steps):.1f}, min: {min(gt_steps)}, max: {max(gt_steps)}")
        print(f"  Pred step — mean: {statistics.mean(pred_steps):.2f}, median: {statistics.median(pred_steps):.1f}, min: {min(pred_steps)}, max: {max(pred_steps)}")
        print(f"  Error (pred-gt) — mean: {statistics.mean(diffs):+.2f}, median: {statistics.median(diffs):+.1f}, min: {min(diffs):+}, max: {max(diffs):+}")


if __name__ == "__main__":
    DIR = "examples/who_and_when/results/your_folder"
    main(DIR)
