"""Convert prepared WebArena traces and ARB annotations to separate JSONL files.

Input is the flat ``cleaned/`` directory from prepare_webarena_data.py and its
``annotations.csv``. A trace's ``task_id`` must equal its filename stem and the
annotation key ``model_name + '_' + task_id``. CSV outcomes must be exactly
Yes/No for looping and side effects, and Successful/Unsuccessful for success
(case insensitive). Invalid or unmatched traces go to ``errors.jsonl``. A
``manifest.json`` is published after all selected traces are processed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from autojudge.jev.contracts import EvaluationCase, EvidenceEvent, ReferenceCase
from autojudge.jev.normalization import parse_finite_float, publish_conversion_manifest

REQUIRED_COLUMNS = {
    "task_id",
    "model_name",
    "trajectory_success",
    "trajectory_side_effect",
    "trajectory_looping",
}


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON literal {value!r}")


def load_annotations(path: Path) -> dict[str, list[tuple[int, dict[str, str | None]]]]:
    """Index annotations by the same composite ID used by prepare_webarena_data.py."""

    annotations: dict[str, list[tuple[int, dict[str, str | None]]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Annotation CSV is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            model_name = row.get("model_name")
            task_id = row.get("task_id")
            if not model_name or not model_name.strip() or not task_id or not task_id.strip():
                continue  # A row without both ID parts cannot be joined to any trace.
            case_id = f"{model_name.strip()}_{task_id.strip()}"
            annotations.setdefault(case_id, []).append((line_number, row))
    return annotations


def _label(row: dict[str, str | None], column: str, options: dict[str, str]) -> str:
    raw = row.get(column)
    value = raw.strip().lower() if isinstance(raw, str) else ""
    if value not in options:
        raise ValueError(f"{column} must be one of {', '.join(options.values())}; got {raw!r}")
    return options[value]


def reference_from_annotation(case_id: str, row: dict[str, str | None]) -> ReferenceCase:
    """Use only the three annotated outcomes; never derive the unannotated optimality."""

    return ReferenceCase(
        case_id=case_id,
        labels={
            "repetition_cycle": _label(row, "trajectory_looping", {"yes": "Yes", "no": "No"}),
            "side_effect": _label(row, "trajectory_side_effect", {"yes": "Yes", "no": "No"}),
            "success": _label(
                row,
                "trajectory_success",
                {"successful": "Successful", "unsuccessful": "Unsuccessful"},
            ),
        },
    )


def case_from_trace(trace: Any, source_file: Path) -> EvaluationCase:
    """Preserve the evidence fields used by the existing WebArena judge, without gold labels."""

    if not isinstance(trace, dict):
        raise ValueError("Trace must be a JSON object")
    case_id = trace.get("task_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("Trace task_id must be a non-empty string")
    if case_id != source_file.stem:
        raise ValueError(f"Trace task_id {case_id!r} differs from filename {source_file.stem!r}")
    goal = trace.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("Trace goal must be a non-empty string")
    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Trace steps must be a non-empty list")

    raw_agent = trace.get("agent")
    agent = raw_agent.strip() if isinstance(raw_agent, str) and raw_agent.strip() else None
    events: list[EvidenceEvent] = []
    for source_step, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"Trace step {source_step} must be a JSON object")
        content = {key: step[key] for key in ("url", "action", "reasoning") if key in step}
        if source_step == len(steps) - 1:
            last_tree = step.get("axtree_pruned") or step.get("axtree")
            if last_tree:
                content["last_accessibility_tree"] = last_tree
        events.append(
            EvidenceEvent(
                id=f"step_{source_step + 1}",
                index=source_step + 1,
                actor=agent,
                content=content,
                source_step=source_step,
            )
        )
    return EvaluationCase(
        case_id=case_id,
        task=goal,
        events=events,
        metadata={"benchmark_id": "webarena", "source_file": source_file.name},
    )


def convert(
    cleaned_dir: Path,
    annotations_file: Path,
    output_dir: Path,
    limit: int | None = None,
) -> dict[str, int]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if not cleaned_dir.is_dir():
        raise FileNotFoundError(f"Cleaned trace directory does not exist: {cleaned_dir}")
    if not annotations_file.is_file():
        raise FileNotFoundError(f"Annotation CSV does not exist: {annotations_file}")

    trace_files = sorted(cleaned_dir.glob("*.json"))
    if not trace_files:
        raise ValueError(f"No JSON traces found in {cleaned_dir}")
    if limit is not None:
        trace_files = trace_files[:limit]
    annotations = load_annotations(annotations_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.jsonl"
    references_path = output_dir / "references.jsonl"
    errors_path = output_dir / "errors.jsonl"
    manifest_path = output_dir / "manifest.json"
    for target in (cases_path, references_path, errors_path, manifest_path):
        if target.exists():
            raise FileExistsError(f"Output already exists: {target}")

    counts = {"cases": 0, "errors": 0}
    seen_ids: set[str] = set()
    with (
        cases_path.open("x", encoding="utf-8") as cases_file,
        references_path.open("x", encoding="utf-8") as references_file,
        errors_path.open("x", encoding="utf-8") as errors_file,
    ):
        for trace_file in trace_files:
            case_id: str | None = None
            annotation_line: int | None = None
            try:
                with trace_file.open(encoding="utf-8") as source:
                    trace = json.load(
                        source,
                        parse_constant=_reject_nonfinite,
                        parse_float=parse_finite_float,
                    )
                case = case_from_trace(trace, trace_file)
                case_id = case.case_id
                if case_id in seen_ids:
                    raise ValueError(f"Duplicate case_id: {case_id}")
                matches = annotations.get(case_id, [])
                if not matches:
                    raise ValueError(f"No annotation matches task_id {case_id!r}")
                if len(matches) != 1:
                    lines = ", ".join(str(line) for line, _ in matches)
                    raise ValueError(f"Ambiguous annotations for task_id {case_id!r} at lines {lines}")
                annotation_line, annotation = matches[0]
                reference = reference_from_annotation(case_id, annotation)
                case_json = json.dumps(
                    case.model_dump(mode="json"), ensure_ascii=False, allow_nan=False
                )
                reference_json = json.dumps(
                    reference.model_dump(mode="json"), ensure_ascii=False, allow_nan=False
                )
            except Exception as exc:  # noqa: BLE001 - keep converting independent traces.
                counts["errors"] += 1
                errors_file.write(
                    json.dumps(
                        {
                            "source_file": trace_file.name,
                            "case_id": case_id,
                            "annotation_line": annotation_line,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue
            seen_ids.add(case_id)
            cases_file.write(case_json + "\n")
            references_file.write(reference_json + "\n")
            counts["cases"] += 1
        for target in (cases_file, references_file, errors_file):
            target.flush()
            os.fsync(target.fileno())
    publish_conversion_manifest(
        output_dir,
        converter="webarena",
        sources=[("annotations", annotations_file)]
        + [("trace", trace_file) for trace_file in trace_files],
        cases=counts["cases"],
        errors=counts["errors"],
        limit=limit,
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cleaned_dir", type=Path, help="Prepared WebArena cleaned/ directory")
    parser.add_argument("annotations_file", type=Path, help="AgentRewardBench annotations.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        counts = convert(args.cleaned_dir, args.annotations_file, args.output_dir, args.limit)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(counts, ensure_ascii=False))
    if counts["errors"] or not counts["cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
