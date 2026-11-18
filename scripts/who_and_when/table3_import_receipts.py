"""Reuse receipts only when inputs, prompts and model settings match."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--target", type=Path, required=True)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("table3_ablation", Path(__file__).with_name("table3_ablation.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = m.p
imports = []
for condition in m.CONDITIONS:
    source, target = args.source / condition, args.target / condition
    old, new = p.read_json(source / "manifest.json"), p.read_json(target / "manifest.json")
    a, b = p.read_json(source / "protocol.json"), p.read_json(target / "protocol.json")
    assert {k: v for k, v in a["config"].items() if k != "concurrency"} == {k: v for k, v in b["config"].items() if k != "concurrency"}
    assert a["code_hashes"] == b["code_hashes"]
    assert p.read_json(source / "prompts.json") == p.read_json(target / "prompts.json")
    assert old["entries"] == new["entries"]
    for file in sorted((source / "calls").glob("*.json")):
        record = p.read_json(file)
        assert record["run_id"] == old["run_id"]
        origin = {"file": str(file.resolve()), "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                  "run_id": record["run_id"], "call_id": record["call_id"]}
        call_id = p.digest([new["run_id"], record["task_id"], record["arm"], record["stage"], record["attempt"]])
        dest = target / "calls" / (call_id + ".json")
        if dest.exists():
            raise ValueError("Destination receipt already exists; refusing overwrite")
        record.update(run_id=new["run_id"], call_id=call_id, imported_from=origin)
        if record.get("stream_file"):
            stream = source / record["stream_file"]
            new_stream = target / "streams" / (call_id + ".sse")
            new_stream.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(stream, new_stream)
            record["stream_file"] = str(new_stream.relative_to(target))
        p.write_json(dest, record)
        imports.append({"condition": condition, "destination": str(dest), "origin": origin})
p.write_json(args.target / "receipt_imports.json", {
    "reason": "Transport-only amendment to concurrency=1 before any trace was completed or scored.",
    "request_hash_validation": "Original client verifies request_sha256 before using every cached receipt.",
    "receipts": imports,
})
print("Imported", len(imports), "receipts; original records retained")
