"""Dataset selection and paired scoring for the Who&When judge comparison."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from statistics import median

REVISION = "59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c"
HASHES = {
    "hand": "d28ffe10b124b2ba46a662be3b438fbe0de7c4c4fcba4f46302c0a4b2575f62f",
    "algo": "1d1720f4a8416ee0ed64bc13fa3f75655891e760a917924a071deac443133cf8",
}
METRICS = ("agent", "step", "joint")


def normalized(question: str) -> str:
    return " ".join(question.casefold().split())


def verify_snapshot(path: Path, split: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[split]:
        raise ValueError(f"Expected the frozen {split} snapshot from revision {REVISION}")


def select_cases(rows: list[dict], hand: list[dict], phase: str) -> list[tuple[dict, str]]:
    if phase == "pilot":
        # Deterministic spread of short and long inputs.
        ordered = sorted(rows, key=lambda r: (len(json.dumps(r["history"])), r["question_ID"]))
        return [(ordered[rank], "hand_dev") for rank in (0, 11, 23, 34, 46, 57)]
    hand_ids = {r["question_ID"] for r in hand}
    hand_questions = {normalized(r["question"]) for r in hand}
    output = []
    for row in rows:
        overlap = row["question_ID"] in hand_ids or normalized(row["question"]) in hand_questions
        output.append((row, "overlap" if overlap else "nonoverlap"))
    if sum(group == "nonoverlap" for _, group in output) != 81:
        raise ValueError("Frozen Who&When snapshots must yield 81 Algo traces without overlap")
    return output


def correct(row: dict | None, reference: dict, field: str) -> bool:
    if row is None or "error" in row:
        return False
    prediction = row.get("prediction", {})
    if field == "joint":
        return all(prediction.get(key) == reference[key] for key in ("agent", "step"))
    return prediction.get(field) == reference[field]


def paired_difference(left: list[bool], right: list[bool], seed: int) -> dict:
    differences = [int(a) - int(b) for a, b in zip(left, right, strict=True)]
    n = len(differences)
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(10_000))
    return {"pp": 100 * sum(differences) / n, "ci95_pp": [samples[249] * 100, samples[9749] * 100]}


def summarize(manifest: dict, results: list[dict]) -> dict:
    indexed = {(r["trace_id"], r["mode"]): r for r in results}
    modes = manifest["modes"]
    cases = manifest["cases"]
    expected = {(case["trace_id"], mode) for case in cases for mode in modes}
    if len(indexed) != len(results) or indexed.keys() - expected:
        raise ValueError("Unexpected or duplicate result rows")
    for case in cases:
        for mode in modes:
            row = indexed.get((case["trace_id"], mode))
            if row and (
                row["reference"] != case["reference"] or row.get("cohort") != case["cohort"]
            ):
                raise ValueError("Result metadata does not match the manifest")
    groups = {"all": cases}
    if manifest["phase"] == "main":
        groups["no_hand_overlap"] = [c for c in cases if c["cohort"] == "nonoverlap"]

    summary = {}
    for group, selected in groups.items():
        arms, outcomes, comparisons = {}, {}, {}
        for mode in modes:
            paired = [(c, indexed.get((c["trace_id"], mode))) for c in selected]
            outcomes[mode] = {
                metric: [correct(r, c["reference"], metric) for c, r in paired]
                for metric in METRICS
            }
            attempts = [r for _, r in paired if r is not None]
            durations = sorted(r["elapsed_seconds"] for r in attempts)
            costs = [r["total_cost_usd"] for r in attempts if r["total_cost_usd"] is not None]
            by_stage = {"meta": [], "specialists": [], "final": []}
            for row in attempts:
                if "meta_usage" in row:
                    by_stage["meta"].append(row["meta_usage"])
                for receipt in row.get("judge_usage", []):
                    by_stage[receipt["stage"]].append(receipt)
            stage_cost = {}
            for stage, receipts in by_stage.items():
                priced = [item["cost_usd"] for item in receipts if item["cost_usd"] is not None]
                stage_cost[stage] = {
                    "known_usd": sum(priced) if priced else None,
                    "priced": len(priced),
                    "records": len(receipts),
                }
            arms[mode] = {
                "n": len(selected),
                "missing": len(selected) - len(attempts),
                "failed": sum("error" in r for r in attempts),
                "accuracy": {k: sum(v) / len(selected) for k, v in outcomes[mode].items()},
                "cost_n": len(costs),
                "known_cost_usd": sum(costs) if costs else None,
                "mean_cost_usd": sum(costs) / len(selected)
                if len(costs) == len(selected)
                else None,
                "recorded_stage_cost": stage_cost,
                "latency_n": len(durations),
                "median_seconds": median(durations) if durations else None,
                "p90_seconds": durations[(9 * len(durations) - 1) // 10] if durations else None,
            }
        complete = all(arm["missing"] == 0 for arm in arms.values())
        if complete:
            for first, second in (("pool", "chat"), ("pool", "direct")):
                if first in outcomes and second in outcomes:
                    comparison = {
                        metric: paired_difference(
                            outcomes[first][metric], outcomes[second][metric], manifest["seed"]
                        )
                        for metric in METRICS
                    }
                    pairs = [
                        (indexed[(c["trace_id"], first)], indexed[(c["trace_id"], second)])
                        for c in selected
                    ]
                    costs = [
                        a["total_cost_usd"] - b["total_cost_usd"]
                        for a, b in pairs
                        if a["total_cost_usd"] is not None and b["total_cost_usd"] is not None
                    ]
                    durations = [
                        a["elapsed_seconds"] - b["elapsed_seconds"]
                        for a, b in pairs
                        if "error" not in a and "error" not in b
                    ]
                    comparison["cost_pairs"] = len(costs)
                    comparison["mean_cost_delta_usd"] = (
                        sum(costs) / len(costs) if len(costs) == len(selected) else None
                    )
                    comparison["successful_latency_pairs"] = len(durations)
                    comparison["median_successful_latency_delta_seconds"] = (
                        median(durations) if durations else None
                    )
                    comparisons[f"{first}_minus_{second}"] = comparison
        summary[group] = {"arms": arms, "comparisons": comparisons}
    return summary
