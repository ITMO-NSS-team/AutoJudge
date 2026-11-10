"""Offline metric reproduction. Python standard library only; no API calls."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def normalize_agent(name):
    return "".join(str(name).casefold().split())

def main():
    checksums = json.loads((HERE / "SHA256SUMS.json").read_text())
    for name, expected in checksums.items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Checksum mismatch: " + name)
    manifest = json.loads((HERE / "manifest.json").read_text())
    rows = list(csv.DictReader((HERE / "references.csv").open()))
    refs = {row["task_id"]: row for row in rows}
    if len(refs) != len(rows) or set(refs) != set(manifest["task_ids"]):
        raise ValueError("Reference manifest mismatch")
    predictions = {}
    for line in (HERE / "predictions.jsonl").read_text().splitlines():
        record = json.loads(line)
        task = record["task_id"]
        if task not in refs or task in predictions:
            raise ValueError("Unknown or duplicate prediction")
        predictions[task] = record["prediction"]
    agent = step = 0
    for task in manifest["task_ids"]:
        ref = refs[task]
        pred = predictions.get(task, {})
        agent += (bool(pred.get("agent")) and
                  normalize_agent(pred["agent"]) == normalize_agent(ref["agent"]))
        try:
            step += int(pred["step"]) == int(ref["step_zero_based"]) + 1
        except (KeyError, TypeError, ValueError):
            pass
    n = len(manifest["task_ids"])
    print(json.dumps({"traces": n, "predictions_present": len(predictions),
        "agent_correct": agent, "agent_accuracy_percent": 100 * agent / n,
        "exact_step_correct": step, "exact_step_accuracy_percent": 100 * step / n}, indent=2))

if __name__ == "__main__":
    main()
