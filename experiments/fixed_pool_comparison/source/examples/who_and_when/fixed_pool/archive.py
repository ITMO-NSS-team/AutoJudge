"""Offline storage audit and a self-contained experiment ZIP, with no credentials."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

from .protocol import ARMS, FILE_HASHES, digest, load_run, read_json, write_json
from .report import save_report
from .runner import call_records

KEY_PATTERN = re.compile(rb"sk-or-v1-[0-9a-fA-F]{64}")


def audit_artifacts(path: Path) -> dict:
    manifest, protocol, _ = load_run(path, check_code=False)
    report = save_report(path)
    issues, traces = [], []
    source_manifest = path / "source_snapshot_manifest.json"
    if not source_manifest.exists():
        issues.append("Missing source snapshot manifest")
    else:
        source_files = read_json(source_manifest)
        for name, expected in source_files.items():
            file = path / "source_snapshot" / name
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                issues.append(f"Missing/changed source snapshot: {name}")
        for name, expected in protocol["code_hashes"].items():
            if source_files.get(name) != expected:
                issues.append(f"Executed-source hash not preserved: {name}")
    for name, expected in FILE_HASHES.items():
        file = path / "dataset_snapshot" / f"{name}.parquet"
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            issues.append(f"Missing/changed public dataset snapshot: {name}")
    for name in ("environment.json", "provenance.json"):
        if not (path / name).is_file():
            issues.append(f"Missing {name}")
    records = call_records(path, manifest["run_id"])
    for record in records:
        if digest(record["request"]) != record["request_sha256"]:
            issues.append(f"Request hash mismatch: {record['call_id']}")
        if "response" not in record:
            issues.append(f"No received API response: {record['call_id']} ({record['state']})")
        if record.get("stream_file") and not (path / record["stream_file"]).is_file():
            issues.append(f"Missing original SSE stream: {record['call_id']}")
        if record.get("cost_usd") is None:
            issues.append(f"Unresolved cost: {record['call_id']}")
    for entry in manifest["entries"]:
        for arm in ARMS:
            task = entry["task_id"]
            file = path / "results" / arm / f"{task}.json"
            status = read_json(file)["status"] if file.exists() else "pending"
            rr = [r for r in records if (r["task_id"], r["arm"]) == (task, arm)]
            validated = {r["stage"] for r in rr if r["state"] == "validated"}
            expected = {f"prepare_{arm}", "overall", *(f"judge_{i}" for i in range(1, 7))}
            if status == "success":
                result = read_json(file)
                if validated != expected or len(result.get("findings", [])) != 6:
                    issues.append(f"Incomplete judge/Overall responses: {arm}/{task}")
                trace_file = path / "pipeline_traces" / arm / f"{task}.json"
                if not trace_file.exists() or len(read_json(trace_file)["node_traces"]) != 7:
                    issues.append(f"Missing native AJ message history: {arm}/{task}")
            traces.append(
                {
                    "task_id": task,
                    "arm": arm,
                    "status": status,
                    "attempts": len(rr),
                    "validated_stages": sorted(validated),
                }
            )
    audit = {
        "run_id": manifest["run_id"],
        "phase": manifest["phase"],
        "execution_complete": report["complete"],
        "storage_ok": not issues,
        "issues": issues,
        "traces": traces,
        "total_attempts": len(records),
        "known_cost_usd": report["accounting_all_attempts"]["known_cost_usd"],
    }
    write_json(path / "artifact_audit.json", audit)
    return audit


def export_archive(path: Path, archive_path: Path) -> dict:
    if archive_path.resolve().is_relative_to(path.resolve()):
        raise ValueError("Write the ZIP outside the run directory")
    audit = audit_artifacts(path)
    (path / "ARCHIVE_README.md").write_text(
        "# AutoJudge controlled-6 / fixed-pool experiment\n\n"
        "Start with report.md and artifact_audit.json. A smoke run verifies execution/storage; "
        "it is not the full benchmark result.\n\n"
        "- manifest.json: selected task IDs and input hashes.\n"
        "- protocol.json / prompts.json: frozen settings and all prompt templates.\n"
        "- inputs/: complete model-visible traces; references.json: separate scoring labels.\n"
        "- calls/: exact request bodies and all returned API response fields for every attempt, "
        "including invalid outputs, provider IDs, usage and cost. No Authorization headers.\n"
        "- streams/: original SSE events, including reasoning fragments, usage and keepalives; "
        "calls/ identifies assembled streaming responses explicitly.\n"
        "- results/: generated roles, prepared evidence, all six findings and Overall verdict.\n"
        "- pipeline_traces/: native AJ message histories, including the actual Overall input.\n"
        "- source_snapshot/: executed code/config/docs; environment.json: dependency versions.\n"
        "- dataset_snapshot/: pinned public WW parquet files for independent verification.\n"
        "- SHA256SUMS.json: checksums of all other included files.\n\n"
        "Reasoning is retained if returned by the provider. Provider-internal information that "
        "was never returned is not available. API credentials and .env are excluded.\n",
        encoding="utf-8",
    )
    top = {
        "manifest.json",
        "protocol.json",
        "prompts.json",
        "references.json",
        "provenance.json",
        "environment.json",
        "source_snapshot_manifest.json",
        "pricing.json",
        "last_session.json",
        "pilot_link.json",
        "report.json",
        "report.md",
        "console.log",
        "artifact_audit.json",
        "ARCHIVE_README.md",
    }
    source_names = (
        set(read_json(path / "source_snapshot_manifest.json"))
        if (path / "source_snapshot_manifest.json").exists()
        else set()
    )
    files = []
    for file in sorted(path.rglob("*")):
        if not file.is_file() or file.is_symlink():
            continue
        rel = file.relative_to(path)
        allowed = (
            str(rel) in top
            or rel.parts[0] in {"calls", "inputs", "results", "pipeline_traces"}
            and file.suffix == ".json"
            or rel.parts[0] == "streams" and file.suffix == ".sse"
            or rel.parts[0] == "source_snapshot"
            and str(Path(*rel.parts[1:])) in source_names
            or rel.parts[0] == "dataset_snapshot"
            and file.name in {f"{s}.parquet" for s in FILE_HASHES}
        )
        if allowed:
            if KEY_PATTERN.search(file.read_bytes()):
                raise ValueError(f"Credential-like content detected in {rel}; archive not created")
            files.append(file)
    sums = {str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    write_json(path / "SHA256SUMS.json", sums)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files + [path / "SHA256SUMS.json"]:
            archive.write(file, "experiment/" + str(file.relative_to(path)))
    # Read back every member rather than merely checking that a ZIP was written.
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
        for name, expected in sums.items():
            if hashlib.sha256(archive.read("experiment/" + name)).hexdigest() != expected:
                raise ValueError(f"Archive checksum mismatch: {name}")
    return audit
