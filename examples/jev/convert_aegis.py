"""Export existing AEGIS launcher evidence and separate actor/error labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autojudge.jev.contracts import EvaluationCase, EvidenceEvent, ReferenceCase
from autojudge.jev.engine import actor_option_id
from autojudge.jev.normalization import aegis_judge_input

from .convert_who_and_when import convert


def normalize_row(row: dict[str, Any]) -> tuple[EvaluationCase, ReferenceCase]:
    task, history = aegis_judge_input(row)
    case = EvaluationCase(
        case_id=row["id"],
        task=task,
        events=[
            EvidenceEvent(
                id=f"message_{index}",
                index=index,
                source_step=message["step"],
                actor=message["agent_name"],
                content=message,
            )
            for index, message in enumerate(history, start=1)
        ],
    )
    pairs = {
        actor_option_id(label["agent_name"], label["error_type"])
        for label in row["ground_truth"]["injected_agents"]
    }
    return case, ReferenceCase(case_id=case.case_id, labels={"actor_error_pairs": sorted(pairs)})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="AEGIS split JSONL or Parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    try:
        counts = convert(
            args.input,
            args.output_dir,
            args.limit,
            normalize_row=normalize_row,
            converter="aegis",
            id_field="id",
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(counts))
    if counts["errors"] or not counts["cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
