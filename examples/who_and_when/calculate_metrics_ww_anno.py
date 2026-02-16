import os
import json
from pathlib import Path

def normalize(s: str) -> str:
    return s.lower().replace(" ", "").replace("_", "")


def main(folder_path: str):
    folder = Path(folder_path)

    agent_correct = 0
    agent_total = 0
    step_correct = 0
    step_total = 0

    for filepath in folder.rglob("*.json"): #all .json 
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            scores = data["summarizer_score"]["scores"]
            for item in scores:
                gt_agent = normalize(item["gt_agent"])
                gt_step = str(item["gt_step"])

                agent = normalize(item["score"]["agent"])
                step = str(item["score"]["step"])

                agent_total += 1
                agent_is_correct = gt_agent in agent or agent in gt_agent
                if agent_is_correct:
                    agent_correct += 1

                step_total += 1
                if gt_step == step:
                    step_correct += 1

        except (KeyError, json.JSONDecodeError) as e:
            print(f"Skeep {filepath}: {type(e).__name__}: {e}")
            continue

    acc_agent = agent_correct / agent_total if agent_total > 0 else 0.0
    acc_step = step_correct / step_total if step_total > 0 else 0.0

    print(f"Accuracy Agent: {acc_agent:.4f} ({agent_correct}/{agent_total})")
    print(f"Accuracy Step: {acc_step:.4f} ({step_correct}/{step_total})")


if __name__ == "__main__":
    DIR = "examples/who_and_when/results/who_and_when_gpt5_mini"
    main(DIR)
