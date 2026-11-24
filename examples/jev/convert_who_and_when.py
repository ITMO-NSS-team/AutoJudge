"""Convert labeled Who&When rows into case/reference JSONL and a completion manifest."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from autojudge.jev.contracts import EvaluationCase, ReferenceCase
from autojudge.jev.normalization import (
    normalize_who_and_when_row,
    parse_finite_float,
    publish_conversion_manifest,
)


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON literal {value!r}")


def source_rows(path: Path, limit: int | None) -> Iterator[tuple[int, Any, str | None]]:
    selected = 0
    if path.suffix == ".parquet":
        from pyarrow import parquet

        source = parquet.ParquetFile(path)
        for batch in source.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                selected += 1
                yield selected, row, None
                if limit is not None and selected >= limit:
                    return
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                selected += 1
                try:
                    raw = json.loads(
                        line,
                        parse_constant=_reject_nonfinite,
                        parse_float=parse_finite_float,
                    )
                    yield line_number, raw, None
                except json.JSONDecodeError as exc:
                    yield line_number, None, f"Invalid JSON: {exc.msg} at column {exc.colno}"
                except ValueError as exc:
                    yield line_number, None, f"Invalid JSON: {exc}"
                if limit is not None and selected >= limit:
                    return
    else:
        raise ValueError("Input must be a .parquet or .jsonl file")

def convert(
    path: Path,
    output_dir: Path,
    limit: int | None = None,
    *,
    normalize_row: Callable[[dict[str, Any]], tuple[EvaluationCase, ReferenceCase]] = normalize_who_and_when_row,
    converter: str = "who_and_when",
    id_field: str = "question_ID",
) -> dict[str, int]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if path.suffix not in {".parquet", ".jsonl"}:
        raise ValueError("Input must be a .parquet or .jsonl file")
    if not path.is_file():
        raise FileNotFoundError(f"Input does not exist: {path}")
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
        for line_number, row, parse_error in source_rows(path, limit):
            case_id = None
            if isinstance(row, dict) and row.get(id_field) is not None:
                case_id = str(row[id_field])
            try:
                if parse_error is not None:
                    raise ValueError(parse_error)
                if not isinstance(row, dict):
                    raise ValueError("Source row must be an object")
                case, reference = normalize_row(row)
                if case.case_id in seen_ids:
                    raise ValueError(f"Duplicate {id_field}: {case.case_id}")
                case_json = json.dumps(
                    case.model_dump(mode="json"), ensure_ascii=False, allow_nan=False
                )
                reference_json = json.dumps(
                    reference.model_dump(mode="json"), ensure_ascii=False, allow_nan=False
                )
            except Exception as exc:  # noqa: BLE001 - preserve other valid source rows.
                counts["errors"] += 1
                errors_file.write(
                    json.dumps(
                        {
                            "source_line": line_number,
                            "case_id": case_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue
            seen_ids.add(case.case_id)
            cases_file.write(case_json + "\n")
            references_file.write(reference_json + "\n")
            counts["cases"] += 1
        for target in (cases_file, references_file, errors_file):
            target.flush()
            os.fsync(target.fileno())
    publish_conversion_manifest(
        output_dir,
        converter=converter,
        sources=[("input", path)],
        cases=counts["cases"],
        errors=counts["errors"],
        limit=limit,
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Original Who&When .parquet or .jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        counts = convert(args.input, args.output_dir, args.limit)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(counts, ensure_ascii=False))
    if counts["errors"] or not counts["cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
