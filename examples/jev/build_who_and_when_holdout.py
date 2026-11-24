"""Build a reproducible Who&When Algo holdout excluding Hand question overlap.

The output ``case_ids.txt`` contains one lexicographically sorted case ID per
line. ``manifest.json`` records the two source hashes and the exact exclusion
rule. Inputs use the same Parquet/JSONL row formats as convert_who_and_when.py.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from autojudge.jev.normalization import (
    normalize_who_and_when_row,
    parse_finite_float,
    sha256_file,
)

NORMALIZATION_RULE = "NFKC, casefold, collapse Unicode whitespace"
EXCLUSION_RULE = "exclude Algo when case_id or normalized question occurs in Hand"


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON literal {value!r}")


def _source_rows(path: Path) -> Iterator[tuple[int, Any]]:
    if path.suffix == ".parquet":
        from pyarrow import parquet

        source = parquet.ParquetFile(path)
        row_number = 0
        for batch in source.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                row_number += 1
                yield row_number, row
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(
                        line,
                        parse_constant=_reject_nonfinite,
                        parse_float=parse_finite_float,
                    )
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                yield line_number, row
    else:
        raise ValueError(f"{path}: input must be .parquet or .jsonl")


def _normalized_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def _load_questions(path: Path) -> dict[str, str]:
    questions: dict[str, str] = {}
    for row_number, row in _source_rows(path):
        try:
            if not isinstance(row, dict):
                raise ValueError("source row must be an object")
            case, _ = normalize_who_and_when_row(row)
            if case.case_id in questions:
                raise ValueError(f"duplicate question_ID {case.case_id!r}")
            questions[case.case_id] = _normalized_question(case.task)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: row {row_number}: {exc}") from exc
    if not questions:
        raise ValueError(f"{path}: no valid source rows")
    return questions


def build_holdout(hand_path: Path, algo_path: Path, output_dir: Path) -> dict[str, int]:
    for path in (hand_path, algo_path):
        if not path.is_file():
            raise FileNotFoundError(f"Input does not exist: {path}")
        if path.suffix not in {".parquet", ".jsonl"}:
            raise ValueError(f"Input must be .parquet or .jsonl: {path}")
    if hand_path.resolve() == algo_path.resolve():
        raise ValueError("Hand and Algo inputs must be different files")

    initial_hashes = {"hand": sha256_file(hand_path), "algo": sha256_file(algo_path)}
    hand = _load_questions(hand_path)
    algo = _load_questions(algo_path)
    final_hashes = {"hand": sha256_file(hand_path), "algo": sha256_file(algo_path)}
    if initial_hashes != final_hashes:
        raise ValueError("An input changed while the holdout was being built")

    hand_questions = set(hand.values())
    selected = sorted(
        case_id
        for case_id, question in algo.items()
        if case_id not in hand and question not in hand_questions
    )
    if not selected:
        raise ValueError("No Algo cases remain after excluding Hand overlap")
    if any("\n" in case_id or "\r" in case_id for case_id in selected):
        raise ValueError("A selected case ID contains a line break; cannot write case_ids.txt")

    counts = {
        "hand": len(hand),
        "algo": len(algo),
        "overlap_by_id_only": sum(
            case_id in hand and question not in hand_questions
            for case_id, question in algo.items()
        ),
        "overlap_by_question_only": sum(
            case_id not in hand and question in hand_questions
            for case_id, question in algo.items()
        ),
        "overlap_by_both": sum(
            case_id in hand and question in hand_questions
            for case_id, question in algo.items()
        ),
        "selected": len(selected),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = output_dir / "case_ids.txt"
    manifest_path = output_dir / "manifest.json"
    if ids_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Holdout output already exists in {output_dir}")

    with ids_path.open("x", encoding="utf-8") as target:
        target.write("\n".join(selected) + "\n")
        target.flush()
        os.fsync(target.fileno())
    manifest = {
        "version": 1,
        "builder": "who_and_when_hand_exclusion",
        "sources": {
            name: {"path": str(path.resolve()), "sha256": final_hashes[name]}
            for name, path in (("hand", hand_path), ("algo", algo_path))
        },
        "selection": {
            "question_normalization": NORMALIZATION_RULE,
            "exclusion": EXCLUSION_RULE,
            "ordering": "lexicographic by case_id",
            "counts": counts,
        },
        "files": {"case_ids.txt": {"sha256": sha256_file(ids_path), "rows": len(selected)}},
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_dir,
            prefix=".manifest-", suffix=".tmp", delete=False,
        ) as target:
            temporary_path = Path(target.name)
            target.write(json.dumps(manifest, ensure_ascii=False, allow_nan=False) + "\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, manifest_path)
        directory_fd = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", type=Path, required=True, help="Hand .parquet or .jsonl")
    parser.add_argument("--algo", type=Path, required=True, help="Algo .parquet or .jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        counts = build_holdout(args.hand, args.algo, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
