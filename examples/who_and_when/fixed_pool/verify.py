"""Verify saved run artifacts offline; optionally recompute the original report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACTS = REPO / "rebuttal/fixed_pool_comparison"
MAIN = "main_20261109_01"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: object) -> str:
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def verify(root: Path, recompute: bool = False) -> dict:
    sums = read(root / "SHA256SUMS.json")
    actual = {str(p.relative_to(root)) for folder in ["runs", "data"] for p in (root / folder).rglob("*") if p.is_file()}
    if set(sums) != actual:
        raise ValueError("Artifact checksum manifest is incomplete")
    for name, expected in sums.items():
        if sha((root / name).read_bytes()) != expected:
            raise ValueError(f"Checksum mismatch: {name}")

    runs = sorted(p.parent for p in (root / "runs").rglob("manifest.json"))
    for run in runs:
        manifest, protocol = read(run / "manifest.json"), read(run / "protocol.json")
        if digest({k: v for k, v in manifest.items() if k != "run_id"}) != manifest["run_id"]:
            raise ValueError(f"Manifest identity mismatch: {run.name}")
        for value, expected in [
            (protocol, manifest["protocol_sha256"]),
            (read(run / "prompts.json"), protocol["prompts_sha256"]),
            (read(run / "references.json"), manifest["references_sha256"]),
        ]:
            if digest(value) != expected:
                raise ValueError(f"Frozen execution data changed: {run.name}")
        for name, expected in manifest["dataset_sha256"].items():
            if sha((root / "data" / (name + ".parquet")).read_bytes()) != expected:
                raise ValueError(f"Dataset changed: {name}")
        for entry in manifest["entries"]:
            if digest(read(run / "inputs" / (entry["task_id"] + ".json"))) != entry["input_sha256"]:
                raise ValueError(f"Input changed: {entry['task_id']}")
        for file in (run / "calls").glob("*.json"):
            call = read(file)
            if call["run_id"] != manifest["run_id"] or file.stem != call["call_id"]:
                raise ValueError(f"Call identity mismatch: {file.name}")
            if digest(call["request"]) != call["request_sha256"]:
                raise ValueError(f"Original request changed: {file.name}")
            if call.get("stream_file") and not (run / call["stream_file"]).is_file():
                raise ValueError(f"Missing original stream: {file.name}")

    main = root / "runs" / MAIN
    manifest, references, report = (read(main / name) for name in ["manifest.json", "references.json", "report.json"])
    calls = [read(p) for p in (main / "calls").glob("*.json")]
    for group, data in report["groups"].items():
        entries = [e for e in manifest["entries"] if group == "all" or e["cohort"] == group]
        ids = {e["task_id"] for e in entries}
        for arm, expected in data["arms"].items():
            correct = {"agent": 0, "step": 0, "joint": 0}
            successful = 0
            for task in ids:
                result = read(main / "results" / arm / (task + ".json"))
                valid = result["status"] == "success"
                successful += valid
                prediction, reference = result.get("prediction", {}), references[task]
                agent = valid and prediction.get("agent") == reference["agent"]
                step = valid and prediction.get("step", 0) - 1 == reference["step_zero_based"]
                for metric, value in [("agent", agent), ("step", step), ("joint", agent and step)]:
                    correct[metric] += int(value)
            if len(ids) != expected["n"] or successful != expected["success"]:
                raise ValueError(f"Trace count mismatch: {group}/{arm}")
            for metric, count in correct.items():
                if count != expected["metrics"][metric]["correct"]:
                    raise ValueError(f"Metric mismatch: {group}/{arm}/{metric}")
            records = [c for c in calls if c["task_id"] in ids and c["arm"] == arm]
            cost = sum(c["cost_usd"] for c in records)
            if len(records) != expected["accounting"]["attempts"] or not math.isclose(cost, expected["accounting"]["total_cost_usd"], abs_tol=1e-10):
                raise ValueError(f"Accounting mismatch: {group}/{arm}")

    if recompute:
        # Import only for optional NumPy-based interval recomputation, without model calls.
        from examples.who_and_when.fixed_pool.protocol import prompt_snapshot
        from examples.who_and_when.fixed_pool.report import build_report

        rebuilt = build_report(main)
        for key, value in report.items():
            if key != "generated_at" and rebuilt[key] != value:
                raise ValueError(f"Recomputed report differs: {key}")
        if prompt_snapshot() != read(main / "prompts.json"):
            raise ValueError("Current code generates different prompts")

    return {"verified_artifacts": len(sums), "runs": len(runs), "main_traces": len(manifest["entries"]), "main_call_records": len(calls), "report_recomputed": recompute, "status": "verified"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--recompute", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.artifacts_dir, arguments.recompute), indent=2))
