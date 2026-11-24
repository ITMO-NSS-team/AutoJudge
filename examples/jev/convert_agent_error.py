"""Join official AgentErrorBench trajectories to their separate step/module labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autojudge.jev.contracts import EvaluationCase, EvidenceEvent, ReferenceCase

from .convert_trail import compact
from .convert_who_and_when import convert


def normalize_row(row):
    messages = row["trace"]["messages"]
    turns = []
    pending = []
    for message in messages:
        pending.append(message)
        if message["role"] == "assistant":
            turns.append(pending)
            pending = []
    budget = max(400, min(10_000, 75_000 // max(len(turns), 1)))
    events = [
        EvidenceEvent(
            id=f"step_{step}",
            index=step,
            actor="agent",
            content={
                "messages": [
                    {"role": m["role"], "content": compact(m.get("content"), budget // len(turn))}
                    for m in turn
                ]
            },
        )
        for step, turn in enumerate(turns, start=1)
    ]
    if pending and events:
        events[-1].content["following_observations"] = [
            {"role": m["role"], "content": compact(m.get("content"), budget // len(pending))}
            for m in pending
        ]
    case = EvaluationCase(
        case_id=row["id"],
        task=compact(messages[0].get("content"), 8_000),
        events=events,
        metadata={"cohort": row["cohort"]},
    )
    module = row["label"]["critical_failure_module"].strip().lower()
    if module == "planning":
        module = "plan"
    step = int(row["label"]["critical_failure_step"])
    return case, ReferenceCase(case_id=case.case_id, labels={"step": step, "module": module})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "source.jsonl"
    with source.open("x", encoding="utf-8") as output:
        for cohort in ("ALFWorld", "GAIA", "WebShop"):
            traces = {
                path.stem: path
                for path in (args.dataset / "Original_Failure_Trajectory" / cohort).glob("*.json")
            }
            labels = json.loads(
                (args.dataset / "Label" / f"{cohort.lower()}_labels.json").read_bytes()
            )
            for label in labels:
                case_id = label["trajectory_id"]
                if case_id not in traces:
                    raise ValueError(f"Missing {cohort} trace {case_id}")
                raw = json.loads(traces[case_id].read_bytes())
                output.write(
                    json.dumps(
                        {"id": case_id, "cohort": cohort, "trace": raw, "label": label},
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
    counts = convert(
        source,
        args.output_dir,
        normalize_row=normalize_row,
        converter="agent_error_bench",
        id_field="id",
    )
    print(json.dumps(counts))
    if counts["errors"] or not counts["cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
