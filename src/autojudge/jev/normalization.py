"""Load normalized cases and adapt labeled Who&When rows to the shared contract."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .contracts import EvaluationCase, EvaluationSpec, EvidenceEvent, ReferenceCase


@dataclass(frozen=True, slots=True)
class NormalizedRow:
    """A valid case or a row-local error; one bad line does not stop loading."""

    line_number: int
    case_id: str | None
    case: EvaluationCase | None
    error: str | None


@dataclass(frozen=True, slots=True)
class ReferenceRow:
    line_number: int
    case_id: str | None
    reference: ReferenceCase | None
    error: str | None


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON literal {value!r}")


def parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite JSON number {value!r}")
    return parsed


def _jsonl_objects(path: str | Path) -> Iterator[tuple[int, Any, str | None]]:

    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
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


def _case_id_from_raw(raw: Any) -> str | None:
    if isinstance(raw, dict) and isinstance(raw.get("case_id"), str):
        return raw["case_id"]
    return None


def load_normalized_jsonl(path: str | Path) -> Iterator[NormalizedRow]:
    """Yield case records and validation failures without reading references."""

    seen_ids: set[str] = set()
    for line_number, raw, parse_error in _jsonl_objects(path):
        case_id = _case_id_from_raw(raw)
        if parse_error is not None:
            yield NormalizedRow(line_number, case_id, None, parse_error)
            continue
        try:
            case = EvaluationCase.model_validate(raw)
        except ValidationError as exc:
            yield NormalizedRow(line_number, case_id, None, str(exc))
            continue
        if case.case_id in seen_ids:
            yield NormalizedRow(line_number, case.case_id, None, "Duplicate case_id")
            continue
        seen_ids.add(case.case_id)
        yield NormalizedRow(line_number, case.case_id, case, None)


def load_references_jsonl(path: str | Path) -> Iterator[ReferenceRow]:
    """Load gold labels separately from cases; malformed rows remain visible."""

    seen_ids: set[str] = set()
    for line_number, raw, parse_error in _jsonl_objects(path):
        case_id = _case_id_from_raw(raw)
        if parse_error is not None:
            yield ReferenceRow(line_number, case_id, None, parse_error)
            continue
        try:
            reference = ReferenceCase.model_validate(raw)
        except ValidationError as exc:
            yield ReferenceRow(line_number, case_id, None, str(exc))
            continue
        if reference.case_id in seen_ids:
            yield ReferenceRow(line_number, reference.case_id, None, "Duplicate case_id")
            continue
        seen_ids.add(reference.case_id)
        yield ReferenceRow(line_number, reference.case_id, reference, None)


def load_spec(path: str | Path) -> EvaluationSpec:
    """Validate one versioned benchmark rubric."""

    with Path(path).open(encoding="utf-8") as source:
        return EvaluationSpec.model_validate(
            json.load(
                source,
                parse_constant=_reject_nonfinite,
                parse_float=parse_finite_float,
            )
        )


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a conversion input or output file's current bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def publish_conversion_manifest(
    output_dir: Path,
    *,
    converter: str,
    sources: list[tuple[str, Path]],
    cases: int,
    errors: int,
    limit: int | None,
) -> None:
    """Publish a completion marker only after all three JSONL files are durable."""

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Output already exists: {manifest_path}")

    rows = {"cases.jsonl": cases, "references.jsonl": cases, "errors.jsonl": errors}
    manifest = {
        "version": 1,
        "converter": converter,
        "sources": [
            {"role": role, "path": str(path.resolve()), "sha256": sha256_file(path)}
            for role, path in sources
        ],
        "selection": {"limit": limit, "selected": cases + errors},
        "counts": {"cases": cases, "references": cases, "errors": errors},
        "files": {
            name: {"sha256": sha256_file(output_dir / name), "rows": count}
            for name, count in rows.items()
        },
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=".manifest-",
            suffix=".tmp",
            delete=False,
        ) as target:
            temporary_path = Path(target.name)
            target.write(json.dumps(manifest, ensure_ascii=False, allow_nan=False) + "\n")
            target.flush()
            os.fsync(target.fileno())
        if manifest_path.exists():
            raise FileExistsError(f"Output already exists: {manifest_path}")
        os.replace(temporary_path, manifest_path)
        directory_fd = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


_WHO_AND_WHEN_REFERENCE_KEYS = frozenset(
    {
        "mistake_agent",
        "mistake_step",
        "mistake_reason",
        "groundtruth",
        "ground_truth",
        "correct_answer",
        "is_correct",
        "is_corrected",
    }
)


def _embedded_reference_key(value: Any) -> str | None:
    """Reject structured annotation fields before a history item becomes judge evidence."""

    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, float) and not math.isfinite(current):
            raise ValueError("Who&When history contains a non-finite number")
        if isinstance(current, dict):
            for key, child in current.items():
                if key in _WHO_AND_WHEN_REFERENCE_KEYS:
                    return key
                pending.append(child)
        elif isinstance(current, list):
            pending.extend(current)
    return None


def normalize_who_and_when_row(row: Mapping[str, Any]) -> tuple[EvaluationCase, ReferenceCase]:
    """Convert one original labeled Who&When row, keeping gold outside the case."""

    history = row["history"]
    if not isinstance(history, list) or not history:
        raise ValueError("Who&When history must be a non-empty list")

    events: list[EvidenceEvent] = []
    for source_step, message in enumerate(history):
        if not isinstance(message, dict):
            raise ValueError(f"Who&When history item {source_step} must be an object")
        reference_key = _embedded_reference_key(message)
        if reference_key is not None:
            raise ValueError(
                f"Who&When history item {source_step} contains reference field {reference_key!r}"
            )
        role = message.get("role") if isinstance(message.get("role"), str) else None
        name = message.get("name") if isinstance(message.get("name"), str) else None
        actor = (name.strip() if name else "") or (
            (role or "").split(" (", 1)[0].strip()
        )
        if not actor:
            raise ValueError(f"Who&When history item {source_step} has no actor")
        source_id = message.get("id")
        events.append(
            EvidenceEvent(
                id=f"message_{source_step + 1}",
                index=source_step + 1,
                actor=actor,
                role=role,
                content=message,
                source_step=source_step,
                source_id=str(source_id) if source_id is not None else None,
            )
        )

    raw_case_id = row["question_ID"]
    if isinstance(raw_case_id, bool) or not isinstance(raw_case_id, (str, int)):
        raise ValueError("Who&When question_ID must be a string or integer")
    case_id = str(raw_case_id).strip()
    if not case_id:
        raise ValueError("Who&When question_ID must be non-empty")
    case = EvaluationCase(
        case_id=case_id,
        task=row["question"],
        events=events,
    )
    raw_step = row["mistake_step"]
    if isinstance(raw_step, bool):
        raise ValueError("Who&When mistake_step must be an integer")
    try:
        source_step = int(raw_step)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Who&When mistake_step must be an integer") from exc
    if source_step != raw_step and str(source_step) != str(raw_step).strip():
        raise ValueError("Who&When mistake_step must be an integer")
    if not 0 <= source_step < len(events):
        raise ValueError("Who&When mistake_step is outside the history")
    reference_agent = row["mistake_agent"]
    if not isinstance(reference_agent, str) or not reference_agent.strip():
        raise ValueError("Who&When mistake_agent must be non-empty")
    reference_agent = reference_agent.strip()
    if events[source_step].actor != reference_agent:
        raise ValueError(
            f"Who&When mistake_agent {reference_agent!r} does not match "
            f"the actor at mistake_step {source_step}: {events[source_step].actor!r}"
        )
    reference = ReferenceCase(
        case_id=case.case_id,
        labels={"agent": reference_agent, "step": source_step + 1},
    )
    return case, reference
