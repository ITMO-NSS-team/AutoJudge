"""Reconstructed Table 3 few-shot ablation using the frozen AutoJudge Q1 engine.

Preparation and reporting are offline. Inference never reads references.json.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/table3"
SOURCE = EXPERIMENT / "source"
sys.path.insert(0, str(SOURCE / "examples/who_and_when"))

from q1 import protocol as p  # noqa: E402
from q1.aj_bridge import evaluate_using_aj  # noqa: E402
from q1.report import accounting as original_accounting  # noqa: E402
from q1.report import paired_interval  # noqa: E402
from q1.runner import HaltRun, OpenRouterClient, StageFailure, call_records  # noqa: E402


class ConservativeClient(OpenRouterClient):
    """Account lost requests at their reservation ceiling, retaining unknown actual cost."""

    def bound_lost(self, record):
        if record.get("state") != "unknown" or record.get("cost_usd") is not None:
            return False
        if record.get("error_type") not in {
            "RemoteProtocolError",
            "SSLError",
            "ConnectError",
            "ReadError",
            "ReadTimeout",
            "WriteError",
            "WriteTimeout",
            "ConnectTimeout",
        }:
            return False
        record.update(
            original_state="unknown",
            state="abandoned",
            actual_cost_usd=None,
            cost_usd=record["reserved_usd"],
            cost_source="reserved_upper_bound_not_actual",
            accounting_note="Response lost. Actual cost remains unknown. Full request reservation debited to budget before any retry.",
        )
        self.save(record)
        return True

    async def preflight(self):
        for record in call_records(self.path, self.manifest["run_id"]):
            self.bound_lost(record)
        await super().preflight()

    async def request(self, payload, task, arm, stage, attempt):
        try:
            return await super().request(payload, task, arm, stage, attempt)
        except HaltRun:
            call_id = p.digest([self.manifest["run_id"], task, arm, stage, attempt])
            file = self.path / "calls" / f"{call_id}.json"
            record = p.read_json(file) if file.exists() else {}
            if not self.bound_lost(record):
                raise
            self.spent += record["cost_usd"]
            self.stopping = self.spent >= self.max_cost
            return record


def accounting(records):
    result = original_accounting(records)
    upper = [r for r in records if r.get("cost_source") == "reserved_upper_bound_not_actual"]
    result["budget_debit_usd"] = result["known_cost_usd"]
    result["known_cost_usd"] -= sum(r["cost_usd"] for r in upper)
    result["upper_bound_only_calls"] = len(upper)
    result["unknown_cost_calls"] += len(upper)
    if upper:
        result["total_cost_usd"] = None
    result["cost_upper_bound_usd"] = result["budget_debit_usd"]
    return result


CONDITIONS = {
    "six": list(p.FIXED_NAMES),
    "four": [p.FIXED_NAMES[i] for i in (0, 1, 2, 5)],
    "completion_complexity": list(p.FIXED_NAMES[:2]),
    "complexity_role_coverage": [p.FIXED_NAMES[1], "MAS_ROLE_COVERAGE_JUDGE"],
    "three": list(p.FIXED_NAMES[:3]),
    "completion_guidelines": [p.FIXED_NAMES[0], "GUIDELINE_CONSISTENCY_JUDGE"],
}
DEFAULT_DATA = Path(
    "/home/alina/Desktop/AutoJudge/examples/who_and_when/results/9bvm-q1/"
    "main_20261109_01/dataset_snapshot/Hand-Crafted.parquet"
)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment_hashes():
    files = [Path(__file__), EXPERIMENT / "reconstructed_examples.json"]
    files += sorted(SOURCE.rglob("*.py"))
    return {str(f.relative_to(ROOT)): file_hash(f) for f in files}


def prepare(path: Path, data: Path):
    import pandas as pd

    if path.exists():
        raise ValueError("Output exists; use run to resume or choose a new output directory")
    if file_hash(data) != p.FILE_HASHES["Hand-Crafted"]:
        raise ValueError("Pinned Hand dataset checksum mismatch")
    rows = pd.read_parquet(data).to_dict(orient="records")
    if len(rows) != 58:
        raise ValueError("Expected 58 Hand development traces")
    base = p.prompt_snapshot()
    examples = {j["name"]: j for j in base["fixed_judges"]}
    reconstructed = p.read_json(EXPERIMENT / "reconstructed_examples.json")
    examples.update({j["name"]: j for j in reconstructed})
    # Preserve the exact original example blocks for the four recovered conditions.
    import re

    blocks = re.split(r"(?=Example \d+ - )", base["few_shot_examples"])
    original_blocks = {
        name: next(b for b in blocks if f'"name": "{name}"' in b) for name in p.FIXED_NAMES
    }
    entries, references, inputs = [], {}, {}
    for row in sorted(rows, key=lambda r: str(r["question_ID"])):
        task = str(row["question_ID"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task) or task in inputs:
            raise ValueError("Unsafe or duplicate task ID")
        inp = p.make_input(row)
        inputs[task] = inp
        entries.append(
            {
                "task_id": task,
                "input_sha256": p.digest(inp),
                "cohort": "hand_dev",
                "messages": len(inp["messages"]),
            }
        )
        references[task] = {
            "agent": str(row["mistake_agent"]),
            "step_zero_based": int(row["mistake_step"]),
        }
    config = p.load_config()
    config["seed"] = 20261118
    config["concurrency"] = 1
    campaign = {
        "version": 1,
        "created_at": p.now(),
        "conditions": CONDITIONS,
        "dataset_revision": p.REVISION,
        "dataset_sha256": file_hash(data),
        "experiment_hashes": experiment_hashes(),
        "repeats": 1,
        "traces_per_condition": 58,
        "planned_calls": 58 * 6 * 8,
        "technical_gate": "First trace in all six conditions must succeed before continuing.",
        "selection": "No configuration selection from these results; Hand is development data.",
        "primary_metric": "Agent accuracy; exact step and joint accuracy are secondary.",
        "inference": "All stages Gemini 2.5 Flash, six generated judges, fixed aggregator; tools disabled.",
        "historical_status": "New controlled reconstruction, not verification of historical Table 3.",
        "reconstructed_roles": [j["name"] for j in reconstructed],
        "lost_request_policy": "Actual cost unknown; debit full conservative reservation and retain provenance before retry. At most three attempts per stage.",
        "concurrency": 1,
        "scoring": p.SCORING,
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "pydantic-ai-slim",
                "pydantic-graph",
                "httpx",
                "numpy",
                "pandas",
                "pyarrow",
                "tiktoken",
                "langfuse",
                "loguru",
                "python-dotenv",
                "genai-prices",
            )
        },
    }
    campaign["campaign_id"] = p.digest(campaign)
    path.mkdir(parents=True)
    p.write_json(path / "campaign.json", campaign)
    shutil.copyfile(Path(__file__), path / "launcher.py")
    shutil.copyfile(
        EXPERIMENT / "reconstructed_examples.json", path / "reconstructed_examples.json"
    )
    (path / "dataset_snapshot").mkdir()
    shutil.copyfile(data, path / "dataset_snapshot/Hand-Crafted.parquet")
    for name, names in CONDITIONS.items():
        selected = []
        for index, role in enumerate(names, 1):
            if role in original_blocks:
                # Only the example number changes; instruction text remains byte-for-byte intact.
                selected.append(re.sub(r"^Example \d+", f"Example {index}", original_blocks[role]))
            else:
                example = {k: v for k, v in examples[role].items() if k != "provenance"}
                selected.append(
                    f"Example {index} - {role}:\n"
                    + json.dumps([example], ensure_ascii=False, indent=2)
                    + "\n"
                )
        fewshot = "\n" + "\n".join(selected)
        prompts = dict(base)
        if base["meta"].count(base["few_shot_examples"]) != 1:
            raise ValueError("Cannot isolate the few-shot slot in the original prompt")
        prompts["meta"] = base["meta"].replace(base["few_shot_examples"], fewshot)
        prompts["few_shot_examples"] = fewshot
        protocol = {
            "version": "table3-reconstruction-1",
            "config": config,
            "scoring": p.SCORING,
            "arms": ["autojudge"],
            "prompts_sha256": p.digest(prompts),
            "code_hashes": p.source_hashes(),
        }
        manifest = {
            "phase": "table3",
            "dataset_revision": p.REVISION,
            "dataset_sha256": {"Hand-Crafted": file_hash(data)},
            "protocol_sha256": p.digest(protocol),
            "references_sha256": p.digest(references),
            "entries": entries,
        }
        manifest["run_id"] = p.digest(manifest)
        sub = path / name
        for filename, value in (
            ("protocol", protocol),
            ("prompts", prompts),
            ("manifest", manifest),
            ("references", references),
        ):
            p.write_json(sub / f"{filename}.json", value)
        for task, inp in inputs.items():
            p.write_json(sub / "inputs" / f"{task}.json", inp)
    print(f"Prepared six conditions × 58 traces = 2784 planned calls: {path}")


def validate_campaign(path):
    campaign = p.read_json(path / "campaign.json")
    if (
        p.digest({k: v for k, v in campaign.items() if k != "campaign_id"})
        != campaign["campaign_id"]
    ):
        raise ValueError("Campaign manifest changed")
    if campaign["experiment_hashes"] != experiment_hashes():
        raise ValueError("Experiment source changed since preparation")
    if campaign["conditions"] != CONDITIONS:
        raise ValueError("Condition definitions changed")
    return campaign


async def run(path, max_cost, env_file, limit):
    from dotenv import dotenv_values

    validate_campaign(path)
    if not math.isfinite(max_cost) or max_cost <= 0:
        raise ValueError("Budget must be positive")
    key = dotenv_values(env_file).get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("No OPENROUTER_API_KEY in specified env file")
    clients = {}
    with (path / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for name in CONDITIONS:
                # Runtime verification checks hashes without reading ground-truth labels.
                sub = path / name
                manifest = p.read_json(sub / "manifest.json")
                protocol, prompts = (
                    p.read_json(sub / "protocol.json"),
                    p.read_json(sub / "prompts.json"),
                )
                if (
                    p.digest({k: v for k, v in manifest.items() if k != "run_id"})
                    != manifest["run_id"]
                ):
                    raise ValueError("Manifest changed")
                if (
                    p.digest(protocol) != manifest["protocol_sha256"]
                    or p.digest(prompts) != protocol["prompts_sha256"]
                ):
                    raise ValueError("Protocol or prompts changed")
                if p.source_hashes() != protocol["code_hashes"]:
                    raise ValueError("AutoJudge source changed")
                client = ConservativeClient(sub, manifest, protocol, key, max_cost)
                clients[name] = (client, prompts)
                await client.preflight()
            entries = next(iter(clients.values()))[0].manifest["entries"]
            for index, entry in enumerate(entries[:limit] if limit else entries):
                # Rotate order to distribute condition ordering without selecting on scores.
                names = list(CONDITIONS)
                names = names[index % 6 :] + names[: index % 6]
                for name in names:
                    client, prompts = clients[name]
                    output = client.path / "results/autojudge" / f"{entry['task_id']}.json"
                    if output.exists():
                        saved = p.read_json(output)
                        if (
                            saved["run_id"] != client.manifest["run_id"]
                            or saved["input_sha256"] != entry["input_sha256"]
                        ):
                            raise ValueError("Result identity mismatch")
                        if index == 0 and saved["status"] != "success":
                            raise HaltRun(
                                "Technical gate failed; inspect before further paid calls"
                            )
                        continue
                    others = sum(c.spent for c, _ in clients.values() if c is not client)
                    client.max_cost = max_cost - others
                    try:
                        result = await evaluate_using_aj(client, entry, "autojudge", prompts)
                    except StageFailure as exc:
                        result = {"status": "failed", "error": str(exc)}
                    result.update(
                        run_id=client.manifest["run_id"],
                        input_sha256=entry["input_sha256"],
                        task_id=entry["task_id"],
                        condition=name,
                        completed_at=p.now(),
                    )
                    p.write_json(output, result)
                    total = sum(c.spent for c, _ in clients.values())
                    print(
                        f"[{index + 1}/58] {name} {entry['task_id']}: {result['status']}; total ${total:.4f}",
                        flush=True,
                    )
                    if index == 0 and result["status"] != "success":
                        raise HaltRun("Technical gate failed; inspect before further paid calls")
                if index == 0:
                    p.write_json(
                        path / "technical_gate.json",
                        {"passed": True, "at": p.now(), "task_id": entry["task_id"]},
                    )
        finally:
            for client, _ in clients.values():
                await client.close()


def report(path):
    campaign = p.read_json(path / "campaign.json")
    results, outcomes, costs = {}, {}, []
    for name in CONDITIONS:
        sub = path / name
        manifest, protocol, _ = p.load_run(sub, check_code=False)
        references = p.read_json(sub / "references.json")
        counts = {k: 0 for k in ("success", "failed", "pending", "agent", "step", "joint")}
        outcomes[name] = []
        for entry in manifest["entries"]:
            file = sub / "results/autojudge" / f"{entry['task_id']}.json"
            result = p.read_json(file) if file.exists() else {"status": "pending"}
            if file.exists() and (
                result["run_id"] != manifest["run_id"]
                or result["input_sha256"] != entry["input_sha256"]
            ):
                raise ValueError("Result identity mismatch")
            valid = result["status"] == "success"
            pred, ref = result.get("prediction", {}), references[entry["task_id"]]
            if valid:
                inp = p.read_json(sub / "inputs" / f"{entry['task_id']}.json")
                p.validate_output("overall", pred, p.schemas(inp)["overall"], inp)
            agent = int(valid and pred.get("agent") == ref["agent"])
            step = int(valid and pred.get("step", 0) - 1 == ref["step_zero_based"])
            counts[result["status"]] += 1
            for metric, value in (("agent", agent), ("step", step), ("joint", agent * step)):
                counts[metric] += value
            outcomes[name].append((agent, step, agent * step))
        records = call_records(sub, manifest["run_id"])
        costs.extend(records)
        results[name] = {
            "n": 58,
            **counts,
            "accuracy_percent": {k: counts[k] / 58 * 100 for k in ("agent", "step", "joint")},
            "accounting": accounting(records),
        }
    complete = all(r["pending"] == 0 for r in results.values())
    comparisons = {}
    if complete:
        for name in CONDITIONS:
            if name != "six":
                comparisons[name + "_minus_six"] = {
                    metric: paired_interval(
                        [a[i] - b[i] for a, b in zip(outcomes[name], outcomes["six"], strict=True)],
                        protocol["config"]["bootstrap_samples"],
                        20261118,
                    )
                    for i, metric in enumerate(("agent", "step", "joint"))
                }
    summary = {
        "campaign_id": campaign["campaign_id"],
        "complete": complete,
        "conditions": results,
        "paired_exploratory_unadjusted": comparisons,
        "accounting": accounting(costs),
        "note": "Hand development data; one generation repeat; two reconstructed examples.",
    }
    p.write_json(path / "report.json", summary)
    lines = [
        "# Reconstructed Table 3",
        "",
        summary["note"],
        "",
        f"Complete: {complete}. Known costs: ${summary['accounting']['known_cost_usd']:.4f}; conservative budget debit: ${summary['accounting']['budget_debit_usd']:.4f}.",
        "",
        "| Few-shot condition | Completed | Agent / 58 (%) | Exact step / 58 (%) | Cost ($) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['success'] + r['failed']}/58 | {r['accuracy_percent']['agent']:.2f} | {r['accuracy_percent']['step']:.2f} | {r['accounting']['known_cost_usd']:.4f} |"
        )
    if not complete:
        lines += [
            "",
            "Incomplete run: displayed fractions include pending traces and are not final accuracy estimates.",
        ]
    (path / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": complete, "accounting": summary["accounting"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report"))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/table3-reconstruction-20261118-olama-v3"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--env-file", type=Path, default=Path("/home/alina/Desktop/AutoJudge/.env"))
    parser.add_argument("--max-cost-usd", type=float, default=100)
    parser.add_argument(
        "--limit",
        type=int,
        help="Technical execution limit, without changing the 58-trace manifest",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.command == "prepare":
        prepare(args.output, args.data)
    elif args.command == "report":
        report(args.output)
    else:
        asyncio.run(run(args.output, args.max_cost_usd, args.env_file, args.limit))


if __name__ == "__main__":
    main()
