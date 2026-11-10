"""Verify the published experiment with the standard library and no network access."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
MAIN = "main_20261109_01/experiment/"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: object) -> str:
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def verify() -> dict:
    with ZipFile(ROOT / "fixed_pool_comparison.zip") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        if archive.testzip() is not None:
            raise ValueError("Archive CRC mismatch")

        def read(name: str) -> dict:
            return json.loads(archive.read(name))

        sums = read("SHA256SUMS.json")
        if set(sums) != set(names) - {"SHA256SUMS.json"}:
            raise ValueError("Archive checksum manifest is incomplete")
        for name, expected in sums.items():
            if sha(archive.read(name)) != expected:
                raise ValueError(f"Checksum mismatch: {name}")

        runs = sorted(n.removesuffix("manifest.json") for n in names if n.endswith("/manifest.json") and "/source_snapshot/" not in n)
        for run in runs:
            manifest, protocol = read(run + "manifest.json"), read(run + "protocol.json")
            if digest({k: v for k, v in manifest.items() if k != "run_id"}) != manifest["run_id"]:
                raise ValueError(f"Manifest identity mismatch: {run}")
            for value, expected in [
                (protocol, manifest["protocol_sha256"]),
                (read(run + "prompts.json"), protocol["prompts_sha256"]),
                (read(run + "references.json"), manifest["references_sha256"]),
            ]:
                if digest(value) != expected:
                    raise ValueError(f"Frozen execution data changed: {run}")
            for entry in manifest["entries"]:
                if digest(read(run + "inputs/" + entry["task_id"] + ".json")) != entry["input_sha256"]:
                    raise ValueError(f"Input changed: {entry['task_id']}")
            sources = read(run + "source_publication.json")["files"]
            if not set(protocol["code_hashes"].values()) <= {v["executed_sha256"] for v in sources.values()}:
                raise ValueError(f"Missing executed source fingerprint: {run}")
            if read(run + "source_snapshot_manifest.json") != {k: v["published_sha256"] for k, v in sources.items()}:
                raise ValueError(f"Source manifest mismatch: {run}")
            for name, fingerprints in sources.items():
                if sha(archive.read(run + "source_snapshot/" + name)) != fingerprints["published_sha256"]:
                    raise ValueError(f"Published source mismatch: {name}")
            for name, expected in read(run + "SHA256SUMS.json").items():
                if sha(archive.read(run + name)) != expected:
                    raise ValueError(f"Run checksum mismatch: {run + name}")

        for name in read(MAIN + "source_publication.json")["files"]:
            if (ROOT / "source" / name).read_bytes() != archive.read(MAIN + "source_snapshot/" + name):
                raise ValueError(f"Repository source differs from archive source: {name}")
        if (ROOT / "source/campaign/launcher.py").read_bytes() != archive.read("campaign/launcher.py"):
            raise ValueError("Launcher source differs")
        for name in ["report.md", "report.json", "source_publication.json"]:
            if (ROOT / name).read_bytes() != archive.read(MAIN + name):
                raise ValueError(f"Repository summary differs from archive: {name}")

        manifest, references, report = (read(MAIN + name) for name in ["manifest.json", "references.json", "report.json"])
        calls = [read(name) for name in names if name.startswith(MAIN + "calls/") and name.endswith(".json")]
        for call in calls:
            if call["run_id"] != manifest["run_id"] or digest(call["request"]) != call["request_sha256"]:
                raise ValueError("Call identity or original request hash mismatch")
        for group, data in report["groups"].items():
            entries = [e for e in manifest["entries"] if group == "all" or e["cohort"] == group]
            ids = {e["task_id"] for e in entries}
            for arm, expected in data["arms"].items():
                correct = {"agent": 0, "step": 0, "joint": 0}
                successful = 0
                for task in ids:
                    result = read(MAIN + f"results/{arm}/{task}.json")
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
        return {"verified_members": len(names), "runs": len(runs), "main_traces": len(manifest["entries"]), "main_call_records": len(calls), "status": "verified"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
