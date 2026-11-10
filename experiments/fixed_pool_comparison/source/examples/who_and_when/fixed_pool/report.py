"""Manifest-based attribution metrics, paired intervals and complete attempt accounting."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .protocol import ARMS, load_run, now, read_json, schemas, validate_output, write_json
from .runner import call_records


def accounting(records: list[dict]) -> dict:
    known = [r["cost_usd"] for r in records if r.get("cost_usd") is not None]
    fields = {}
    for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens"):
        values = [r[field] for r in records if type(r.get(field)) is int]
        fields[field] = sum(values)
        fields[field + "_unknown_calls"] = len(records) - len(values)
    return {
        "attempts": len(records),
        "known_cost_usd": sum(known),
        "unknown_cost_calls": len(records) - len(known),
        "manual_cost_calls": sum(r.get("cost_source") == "manual_reconciliation" for r in records),
        "pre_generation_rejections": sum(
            r.get("cost_source") == "pre_generation_http_rejection" for r in records
        ),
        "total_cost_usd": sum(known) if len(known) == len(records) else None,
        "providers": sorted({r["provider"] for r in records if r.get("provider")}),
        "response_models": sorted(
            {r["response_model"] for r in records if r.get("response_model")}
        ),
        "duration_seconds_sum": sum(r.get("duration_seconds", 0) for r in records),
        **fields,
    }


def paired_interval(differences: list[int], samples: int, seed: int) -> dict:
    values = np.asarray(differences, dtype=float)
    if not len(values):
        return {"delta_pp": None, "ci95_pp": None}
    rng = np.random.default_rng(seed)
    # Batched resampling keeps the report cheap even when bootstrap_samples is increased.
    means = []
    for start in range(0, samples, 1000):
        indices = rng.integers(0, len(values), size=(min(1000, samples - start), len(values)))
        means.extend(values[indices].mean(axis=1))
    interval = np.percentile(means, [2.5, 97.5]) * 100
    return {
        "delta_pp": float(values.mean() * 100),
        "ci95_pp": interval.tolist(),
        "paired_discordant_traces": int(np.count_nonzero(values)),
    }


def build_report(path: Path) -> dict:
    manifest, protocol, _ = load_run(path, check_code=False)
    references = read_json(path / "references.json")
    records = call_records(path, manifest["run_id"])
    allowed_ids = {e["task_id"] for e in manifest["entries"]}
    if any(r["task_id"] not in allowed_ids or r["arm"] not in ARMS for r in records):
        raise ValueError("Calls outside the manifest cannot be included in this report")
    outcomes = {}
    for entry in manifest["entries"]:
        task = entry["task_id"]
        inp = read_json(path / "inputs" / f"{task}.json")
        for arm in ARMS:
            file = path / "results" / arm / f"{task}.json"
            result = read_json(file) if file.exists() else {"status": "pending"}
            if file.exists() and (
                result["run_id"] != manifest["run_id"]
                or result["input_sha256"] != entry["input_sha256"]
            ):
                raise ValueError("Result identity does not match manifest")
            valid = result["status"] == "success"
            if valid:
                validate_output("overall", result["prediction"], schemas(inp)["overall"], inp)
            ref = references[task]
            pred = result.get("prediction", {})
            agent = valid and pred.get("agent") == ref["agent"]
            step = valid and pred.get("step", 0) - 1 == ref["step_zero_based"]
            outcomes[task, arm] = {
                "status": result["status"],
                "agent": int(agent),
                "step": int(step),
                "joint": int(agent and step),
            }
    groups = {"all": manifest["entries"]}
    if manifest["phase"] == "main":
        for cohort in ("nonoverlap", "overlap"):
            groups[cohort] = [e for e in manifest["entries"] if e["cohort"] == cohort]
    reports = {}
    for name, entries in groups.items():
        ids = {e["task_id"] for e in entries}
        arms = {}
        for arm in ARMS:
            outs = [outcomes[e["task_id"], arm] for e in entries]
            counts = {
                status: sum(o["status"] == status for o in outs)
                for status in ("success", "failed", "pending")
            }
            if sum(counts.values()) != len(entries):
                raise ValueError("Unknown result status")
            stage_records = [r for r in records if r["arm"] == arm and r["task_id"] in ids]
            by_stage = defaultdict(list)
            for record in stage_records:
                category = (
                    "prepare"
                    if record["stage"].startswith("prepare_")
                    else ("judge" if record["stage"].startswith("judge_") else "overall")
                )
                by_stage[category].append(record)
            arms[arm] = {
                "n": len(entries),
                **counts,
                "coverage": counts["success"] / len(entries) if entries else 0,
                "metrics": {
                    metric: {
                        "correct": sum(o[metric] for o in outs),
                        "accuracy_percent": 100 * sum(o[metric] for o in outs) / len(entries)
                        if entries
                        else None,
                    }
                    for metric in ("agent", "step", "joint")
                },
                "accounting": accounting(stage_records),
                "stages": {stage: accounting(rr) for stage, rr in by_stage.items()},
            }
        deltas = {
            metric: paired_interval(
                [
                    outcomes[e["task_id"], "autojudge"][metric]
                    - outcomes[e["task_id"], "fixed"][metric]
                    for e in entries
                ],
                protocol["config"]["bootstrap_samples"],
                protocol["config"]["seed"],
            )
            for metric in ("agent", "step", "joint")
        }
        a, f = (arms[arm]["accounting"]["total_cost_usd"] for arm in ARMS)
        reports[name] = {
            "arms": arms,
            "autojudge_minus_fixed": deltas,
            "cost_ratio_autojudge_over_fixed": a / f if a is not None and f else None,
        }
    complete = all(o["status"] != "pending" for o in outcomes.values())
    return {
        "run_id": manifest["run_id"],
        "phase": manifest["phase"],
        "generated_at": now(),
        "complete": complete,
        "provisional": not complete,
        "scoring": protocol["scoring"],
        "budget_matching": "Not assumed: report actual cost ratio and token usage",
        "bootstrap": {
            "method": "paired trace bootstrap, percentile 95% CI",
            "samples": protocol["config"]["bootstrap_samples"],
            "seed": protocol["config"]["seed"],
        },
        "reference_annotation_out_of_range": [
            t for t, r in references.items() if r["annotation_out_of_range"]
        ],
        "accounting_all_attempts": accounting(records),
        "groups": reports,
    }


def save_report(path: Path) -> dict:
    report = build_report(path)
    write_json(path / "report.json", report)
    lines = [
        "# WW controlled-6 vs strong fixed pool",
        "",
        f"Phase: {report['phase']}. Complete: {report['complete']}.",
        "",
    ]
    if report["provisional"]:
        lines += ["**INCOMPLETE / PROVISIONAL: pending traces remain in the denominator.**", ""]
    for group, data in report["groups"].items():
        lines += [
            f"## {group}",
            "",
            "| Method | N | Agent Acc | Exact Step Acc | Joint | Coverage | Input tokens | Output tokens (incl. reasoning) | Known cost | Unknown-cost calls |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for arm, a in data["arms"].items():

            def metric(m):
                return f"{a['metrics'][m]['accuracy_percent']:.2f}%" if a["n"] else "n/a"

            c = a["accounting"]
            lines.append(
                f"| {arm} | {a['n']} | {metric('agent')} | {metric('step')} | "
                f"{metric('joint')} | {a['coverage']:.1%} | {c['prompt_tokens']} | "
                f"{c['completion_tokens']} | ${c['known_cost_usd']:.6f} | {c['unknown_cost_calls']} |"
            )
        lines += ["", "AJ − fixed (percentage points; paired 95% CI):", ""]
        for metric, delta in data["autojudge_minus_fixed"].items():
            if delta["delta_pp"] is not None:
                lo, hi = delta["ci95_pp"]
                lines.append(f"- {metric}: {delta['delta_pp']:+.2f} pp [{lo:+.2f}, {hi:+.2f}]")
        ratio = data["cost_ratio_autojudge_over_fixed"]
        lines += [
            "",
            f"Cost ratio AJ/fixed: {ratio:.3f}."
            if ratio is not None
            else "Cost ratio: unavailable.",
            "",
        ]
    lines += [
        "Costs include all saved attempts, including failed/invalid responses. Reasoning is a",
        "subset of output tokens and is not billed twice here. Unknown token counts are shown",
        "in report.json; displayed token totals sum known counts only. Manual billing",
        "reconciliations and pre-generation HTTP rejections are separately counted there.",
        "Do not infer strict budget matching from equal call counts.",
        "",
    ]
    (path / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report
