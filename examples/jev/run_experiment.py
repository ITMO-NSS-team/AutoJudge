"""Run paired Jev direct, fixed-plan, and adaptive-plan benchmark experiments.

Cases and references are separate normalized JSONL files. The planner and Jev
evaluator see only cases and the public evaluation spec; labels enter at scoring.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import inspect
import json
import math
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator
from urllib.parse import urlsplit

from autojudge.jev.client import JEV_MODEL, JevClient
from autojudge.jev.contracts import (
    EvaluationCase,
    EvaluationSpec,
    JudgePlan,
    ReferenceCase,
)
from autojudge.jev.engine import JevEvaluator
from autojudge.jev.normalization import (
    load_normalized_jsonl,
    load_references_jsonl,
    load_spec,
    parse_finite_float,
)
from autojudge.jev.planner import PROMPT_VERSION, JudgePlanGenerator
from autojudge.jev.preflight import validate_case_against_spec
from autojudge.jev.scoring import score_case, summarize_runs, validate_reference
from autojudge.model_connection import (
    OPENROUTER_DECISIONS_URL,
    chat_api_key,
    decision_connection,
    model_http_client,
    omniroute_base_url,
)

MODES = ("direct", "fixed", "adaptive")
MODE_ORDER_POLICY = "cyclic_by_case_v1"
SOURCE_FILES = (
    "pyproject.toml",
    "examples/jev/run_experiment.py",
    "examples/jev/build_who_and_when_holdout.py",
    "examples/jev/convert_who_and_when.py",
    "examples/jev/convert_webarena.py",
    "src/autojudge/jev/client.py",
    "src/autojudge/jev/contracts.py",
    "src/autojudge/jev/engine.py",
    "src/autojudge/jev/planner.py",
    "src/autojudge/jev/preflight.py",
    "src/autojudge/jev/scoring.py",
    "src/autojudge/jev/normalization.py",
    "src/autojudge/meta_agents/base.py",
    "src/autojudge/model_connection.py",
    "src/autojudge/pipeline/types.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {name: _sha256(root / name) for name in SOURCE_FILES}


def _default_parameter(cls: type[Any], name: str) -> Any:
    parameter = inspect.signature(cls.__init__).parameters[name]
    if parameter.default is inspect.Parameter.empty:
        raise ValueError(f"{cls.__name__}.{name} has no default")
    return parameter.default


def _resolved_runtime(args: argparse.Namespace, modes: list[str]) -> dict[str, Any]:
    return {
        "jev": {
            "max_attempts": args.max_attempts,
            "max_concurrent_requests": args.max_concurrent_requests,
            "batch_size": args.batch_size,
            "max_choices": _default_parameter(JevEvaluator, "max_choices"),
            "shortlist_per_window": _default_parameter(
                JevEvaluator, "shortlist_per_window"
            ),
            "max_binary_candidates": _default_parameter(
                JevEvaluator, "max_binary_candidates"
            ),
            "max_state_chars": _default_parameter(JevEvaluator, "max_state_chars"),
            "retry_delay_seconds": _default_parameter(
                JevClient, "retry_delay_seconds"
            ),
            "max_retry_delay_seconds": _default_parameter(
                JevClient, "max_retry_delay_seconds"
            ),
        },
        "meta": (
            {
                "temperature": args.meta_temperature,
                "specialist_count": args.specialist_count,
                "max_tokens": _default_parameter(JudgePlanGenerator, "max_tokens"),
                "timeout": _default_parameter(JudgePlanGenerator, "timeout"),
                "retries": _default_parameter(JudgePlanGenerator, "retries"),
            }
            if "adaptive" in modes
            else None
        ),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        )
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def _sync_directory(path: Path) -> None:
    """Persist a newly published filename before relying on it for recovery."""

    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@contextmanager
def _exclusive_run_lock(output_dir: Path) -> Iterator[None]:
    """Hold a POSIX advisory lock throughout one ledger run or resume."""

    with (output_dir / ".run.lock").open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Output directory is already running: {output_dir}") from exc
        yield


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"Non-finite JSON literal {value!r}")


def _verify_conversion_manifest(
    cases_path: Path,
    references_path: Path,
    *,
    allow_unmanifested: bool,
) -> Path | None:
    """Reject a converter output that was interrupted before its final manifest."""
    if cases_path.parent.resolve() != references_path.parent.resolve():
        if allow_unmanifested:
            return None
        raise ValueError(
            "Cases and references must share a converter directory; "
            "use --allow-unmanifested-input for manually prepared files"
        )
    manifest_path = cases_path.parent / "manifest.json"
    if not manifest_path.is_file():
        if allow_unmanifested:
            return None
        raise ValueError(
            f"Missing conversion manifest {manifest_path}; conversion may be incomplete. "
            "Use --allow-unmanifested-input only for manually verified inputs"
        )
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
            parse_float=parse_finite_float,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid conversion manifest {manifest_path}: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("version")) is not int
        or manifest["version"] != 1
    ):
        raise ValueError("Conversion manifest must be a version 1 JSON object")
    counts = manifest.get("counts")
    files = manifest.get("files")
    if not isinstance(counts, dict) or not isinstance(files, dict):
        raise ValueError("Conversion manifest has no counts or files")
    if cases_path.name != "cases.jsonl" or references_path.name != "references.jsonl":
        raise ValueError("Manifested inputs must be named cases.jsonl and references.jsonl")
    expected_paths = {
        "cases.jsonl": cases_path,
        "references.jsonl": references_path,
        "errors.jsonl": cases_path.parent / "errors.jsonl",
    }
    for name, path in expected_paths.items():
        entry = files.get(name)
        if not isinstance(entry, dict) or not path.is_file():
            raise ValueError(f"Conversion manifest is missing {name}")
        if entry.get("sha256") != _sha256(path):
            raise ValueError(f"Conversion manifest hash differs for {name}")
        with path.open(encoding="utf-8") as source:
            row_count = sum(bool(line.strip()) for line in source)
        if type(entry.get("rows")) is not int or entry["rows"] != row_count:
            raise ValueError(f"Conversion manifest row count differs for {name}")
    for count_name, filename in (
        ("cases", "cases.jsonl"),
        ("references", "references.jsonl"),
        ("errors", "errors.jsonl"),
    ):
        if (
            type(counts.get(count_name)) is not int
            or counts[count_name] != files[filename]["rows"]
        ):
            raise ValueError(f"Conversion manifest count differs for {count_name}")
    if counts["cases"] != counts["references"]:
        raise ValueError("Conversion cases and references have different counts")
    selection = manifest.get("selection")
    if (
        not isinstance(selection, dict)
        or type(selection.get("selected")) is not int
        or selection["selected"] != counts["cases"] + counts["errors"]
    ):
        raise ValueError("Conversion manifest selected count is inconsistent")
    if counts["errors"]:
        raise ValueError(
            f"Conversion produced {counts['errors']} errors; inspect errors.jsonl first"
        )
    return manifest_path


def _load_inputs(
    cases_path: Path,
    references_path: Path,
    spec_path: Path,
    limit: int | None,
    *,
    case_ids_file: Path | None = None,
    allow_unmanifested: bool = False,
) -> tuple[EvaluationSpec, list[EvaluationCase], dict[str, ReferenceCase]]:
    _verify_conversion_manifest(
        cases_path, references_path, allow_unmanifested=allow_unmanifested
    )
    spec = load_spec(spec_path)
    case_rows = list(load_normalized_jsonl(cases_path))
    reference_rows = list(load_references_jsonl(references_path))
    problems = [
        f"cases line {row.line_number}: {row.error}"
        for row in case_rows
        if row.error is not None
    ] + [
        f"references line {row.line_number}: {row.error}"
        for row in reference_rows
        if row.error is not None
    ]
    if problems:
        preview = "; ".join(problems[:5])
        suffix = f"; {len(problems) - 5} more" if len(problems) > 5 else ""
        raise ValueError(f"Invalid normalized input: {preview}{suffix}")

    cases = [row.case for row in case_rows if row.case is not None]
    references = {
        row.reference.case_id: row.reference
        for row in reference_rows
        if row.reference is not None
    }
    if not cases:
        raise ValueError("No normalized cases found")
    all_case_ids = {case.case_id for case in cases}
    missing = all_case_ids - set(references)
    extra = set(references) - all_case_ids
    if missing or extra:
        raise ValueError(
            f"Case/reference IDs do not match: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if case_ids_file is not None:
        selected_ids: list[str] = []
        seen_selected: set[str] = set()
        for line_number, line in enumerate(
            case_ids_file.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            case_id = line.strip()
            if not case_id:
                raise ValueError(f"Empty case ID at {case_ids_file}:{line_number}")
            if case_id in seen_selected:
                raise ValueError(f"Duplicate case ID {case_id!r} in {case_ids_file}")
            seen_selected.add(case_id)
            selected_ids.append(case_id)
        if not selected_ids:
            raise ValueError(f"Case ID selection is empty: {case_ids_file}")
        unknown = seen_selected - all_case_ids
        if unknown:
            raise ValueError(f"Selected case IDs are absent from cases: {sorted(unknown)}")
        cases_by_id = {case.case_id: case for case in cases}
        cases = [cases_by_id[case_id] for case_id in selected_ids]
    elif limit is not None:
        cases = cases[:limit]
    selected_references = {case.case_id: references[case.case_id] for case in cases}
    for case in cases:
        reference = selected_references[case.case_id]
        validate_reference(spec, reference)
        validate_case_against_spec(case, spec, reference)
    return spec, cases, selected_references


def _preflight(
    args: argparse.Namespace,
) -> tuple[
    EvaluationSpec,
    list[EvaluationCase],
    dict[str, ReferenceCase],
    list[str],
    JudgePlan | None,
]:
    if args.case_ids_file is not None and args.limit is not None:
        raise ValueError("--case-ids-file and --limit cannot be combined")
    spec, cases, references = _load_inputs(
        args.cases,
        args.references,
        args.spec,
        args.limit,
        case_ids_file=args.case_ids_file,
        allow_unmanifested=args.allow_unmanifested_input,
    )
    metric_ids = {metric.id for metric in spec.metrics}
    if args.primary_metric not in metric_ids:
        raise ValueError(
            f"--primary-metric {args.primary_metric!r} is not in the spec: "
            f"{sorted(metric_ids)}"
        )
    modes = list(args.modes)
    if not modes:
        raise ValueError("At least one mode is required")
    if len(set(modes)) != len(modes):
        raise ValueError("Modes must be unique")
    if "fixed" in modes and args.fixed_plan is None:
        raise ValueError("Fixed mode requires --fixed-plan")
    if args.fixed_plan is not None and "fixed" not in modes:
        raise ValueError("--fixed-plan requires fixed mode")

    fixed_plan = (
        JudgePlan.model_validate_json(args.fixed_plan.read_text(encoding="utf-8"))
        if args.fixed_plan is not None
        else None
    )
    minimum = max(2, len(spec.targets))
    maximum = max(6, len(spec.targets) + 2)
    if fixed_plan is not None:
        planned = {item.target_id for item in fixed_plan.specialists}
        expected = {target.id for target in spec.targets}
        if planned != expected:
            raise ValueError(
                f"Fixed plan target mismatch: missing={sorted(expected - planned)}, "
                f"unknown={sorted(planned - expected)}"
            )
        if not minimum <= len(fixed_plan.specialists) <= maximum:
            raise ValueError(
                f"Fixed plan must have [{minimum}, {maximum}] specialists"
            )
    if args.specialist_count is not None and not {"fixed", "adaptive"}.intersection(modes):
        raise ValueError("--specialist-count requires fixed or adaptive mode")
    if fixed_plan is not None and "adaptive" in modes and args.specialist_count is None:
        # A fixed-vs-adaptive comparison should use the same specialist budget.
        args.specialist_count = len(fixed_plan.specialists)
    if args.specialist_count is not None:
        if not minimum <= args.specialist_count <= maximum:
            raise ValueError(
                f"--specialist-count must be in [{minimum}, {maximum}] for this spec"
            )
        if fixed_plan is not None and len(fixed_plan.specialists) != args.specialist_count:
            raise ValueError(
                f"Fixed plan has {len(fixed_plan.specialists)} specialists; "
                f"expected {args.specialist_count}"
            )
    return spec, cases, references, modes, fixed_plan


def _endpoint_fingerprint(url: str) -> dict[str, Any]:
    """Identify a route without persisting URL credentials or query parameters."""

    parsed = urlsplit(url)
    return {
        "scheme": parsed.scheme or None,
        "host": parsed.hostname,
        "sha256": hashlib.sha256(url.encode("utf-8")).hexdigest() if url else None,
        "public_url": url if url == OPENROUTER_DECISIONS_URL else None,
    }


def _pydantic_ai_version() -> str | None:
    for distribution in ("pydantic-ai-slim", "pydantic-ai"):
        try:
            return version(distribution)
        except PackageNotFoundError:
            continue
    return None


def _dependency_versions() -> dict[str, str | None]:
    """Record installed libraries whose behavior or prices affect a run."""

    installed: dict[str, str | None] = {}
    for distribution in ("pydantic", "httpx", "genai-prices"):
        try:
            installed[distribution] = version(distribution)
        except PackageNotFoundError:
            installed[distribution] = None
    installed["pydantic-ai"] = _pydantic_ai_version()
    return installed


def _route_manifest(modes: list[str]) -> dict[str, Any]:
    """Mirror the local route selection without checking credentials or opening a client."""

    omni_base = omniroute_base_url()
    direct_decisions = os.getenv("OPENROUTER_DECISIONS_URL", "").strip()
    if direct_decisions:
        decisions_provider = "openrouter"
        decisions_url = direct_decisions
    elif omni_base:
        decisions_provider = "omniroute"
        decisions_url = os.getenv("OMNIROUTE_DECISIONS_URL", "").strip()
    else:
        decisions_provider = "openrouter"
        decisions_url = OPENROUTER_DECISIONS_URL

    meta_route: dict[str, Any] | None = None
    if "adaptive" in modes:
        meta_route = (
            {
                "provider": "OpenAIProvider",
                "endpoint": _endpoint_fingerprint(omni_base),
            }
            if omni_base
            else {
                "provider": "OpenRouterProvider",
                "endpoint": "provider_default",
            }
        )
    return {
        "version": 1,
        "meta_provider_package_version": _pydantic_ai_version() if meta_route else None,
        "decisions": {
            "provider": decisions_provider,
            "endpoint": _endpoint_fingerprint(decisions_url),
        },
        "meta": meta_route,
    }


def _input_manifest(args: argparse.Namespace) -> dict[str, Any]:
    conversion_manifest = args.cases.parent / "manifest.json"
    case_ids_manifest = (
        args.case_ids_file.parent / "manifest.json"
        if args.case_ids_file is not None
        else None
    )
    return {
        "cases": {"path": str(args.cases.resolve()), "sha256": _sha256(args.cases)},
        "references": {
            "path": str(args.references.resolve()),
            "sha256": _sha256(args.references),
        },
        "spec": {"path": str(args.spec.resolve()), "sha256": _sha256(args.spec)},
        "case_ids_file": (
            {"path": str(args.case_ids_file.resolve()), "sha256": _sha256(args.case_ids_file)}
            if args.case_ids_file is not None
            else None
        ),
        "case_ids_manifest": (
            {"path": str(case_ids_manifest.resolve()), "sha256": _sha256(case_ids_manifest)}
            if case_ids_manifest is not None and case_ids_manifest.is_file()
            else None
        ),
        "fixed_plan": (
            {"path": str(args.fixed_plan.resolve()), "sha256": _sha256(args.fixed_plan)}
            if args.fixed_plan is not None
            else None
        ),
        "conversion_manifest": (
            {"path": str(conversion_manifest.resolve()), "sha256": _sha256(conversion_manifest)}
            if conversion_manifest.is_file()
            else None
        ),
    }


def _validate_resume_manifest(
    actual: dict[str, Any], expected: dict[str, Any]
) -> None:
    for key in (
        "benchmark_id",
        "spec_version",
        "primary_metric",
        "case_ids",
        "modes",
        "models",
        "planner_prompt_version",
        "runtime",
        "routes",
        "dependency_versions",
        "execution_order",
    ):
        if actual.get(key) != expected[key]:
            raise ValueError(f"Cannot resume: manifest {key!r} differs from this run")
    actual_sources = actual.get("source_sha256")
    expected_sources = expected["source_sha256"]
    if not isinstance(actual_sources, dict):
        raise ValueError("Cannot resume: manifest has no implementation hashes")
    changed_sources = sorted(
        set(actual_sources) ^ set(expected_sources)
        | {
            name
            for name, digest in expected_sources.items()
            if actual_sources.get(name) != digest
        }
    )
    if changed_sources:
        raise ValueError(
            f"Cannot resume: implementation SHA-256 differs for {changed_sources}"
        )
    actual_inputs = actual.get("inputs")
    if not isinstance(actual_inputs, dict):
        raise ValueError("Cannot resume: manifest has no input hashes")
    for name, expected_input in expected["inputs"].items():
        actual_input = actual_inputs.get(name)
        actual_hash = actual_input.get("sha256") if isinstance(actual_input, dict) else None
        expected_hash = expected_input["sha256"] if expected_input is not None else None
        if actual_hash != expected_hash:
            raise ValueError(f"Cannot resume: {name} SHA-256 differs from the manifest")


def _summary_record(
    record: dict[str, Any],
    spec: EvaluationSpec,
    case: EvaluationCase,
    reference: ReferenceCase,
) -> dict[str, Any]:
    required = ("case_id", "mode", "status", "scores", "stages")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Incomplete result record: missing {missing}")
    if not isinstance(record["status"], str) or record["status"] not in {
        "ok",
        "partial",
        "error",
    }:
        raise ValueError("Invalid result status")
    if record.get("reference") != reference.labels:
        raise ValueError("Stored reference differs from the current input")
    if not isinstance(record["scores"], dict) or record["scores"] != score_case(
        spec, case, reference, record
    ):
        raise ValueError("Stored scores differ from answers and reference")
    stages = record["stages"]
    if not isinstance(stages, dict):
        raise ValueError("Invalid result stages")
    for name in ("planner", "judge", "total"):
        stage = stages.get(name)
        if not isinstance(stage, dict):
            raise ValueError(f"Missing {name} stage")
        for key in ("elapsed_seconds", "recorded_cost_usd"):
            value = stage.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"Invalid {name} stage {key}")
        cost = stage.get("cost_usd")
        complete = stage.get("cost_complete")
        if not isinstance(complete, bool):
            raise ValueError(f"Invalid {name} stage cost_complete")
        if complete:
            if (
                isinstance(cost, bool)
                or not isinstance(cost, (int, float))
                or not math.isfinite(cost)
                or cost < 0
            ):
                raise ValueError(f"Invalid {name} stage cost")
        elif cost is not None:
            raise ValueError(f"Invalid {name} stage cost")
    calls = record.get("calls")
    if not isinstance(calls, list):
        raise ValueError("Invalid result calls")
    jev_calls = http_attempts = retries = 0
    for call in calls:
        attempts = call.get("attempts") if isinstance(call, dict) else None
        if not isinstance(attempts, list):
            raise ValueError("Invalid Jev call attempts")
        if any(
            not isinstance(attempt, dict)
            or type(attempt.get("number")) is not int
            or attempt["number"] != number
            for number, attempt in enumerate(attempts, start=1)
        ):
            raise ValueError("Invalid Jev attempt sequence")
        if attempts:
            jev_calls += 1
            http_attempts += len(attempts)
            retries += len(attempts) - 1
    summary = {key: record[key] for key in required}
    summary.update(
        {"jev_calls": jev_calls, "http_attempts": http_attempts, "retries": retries}
    )
    return summary


def _ordered_pairs(
    cases: list[EvaluationCase], modes: list[str]
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for case_index, case in enumerate(cases):
        offset = case_index % len(modes)
        pairs.extend((case.case_id, mode) for mode in modes[offset:] + modes[:offset])
    return pairs


def _load_existing_records(
    path: Path,
    spec: EvaluationSpec,
    cases: dict[str, EvaluationCase],
    references: dict[str, ReferenceCase],
    modes: list[str],
    expected_order: list[tuple[str, str]],
) -> tuple[list[dict[str, Any]], set[tuple[str, str]], tuple[int, bytes] | None]:
    if not path.exists():
        return [], set(), None
    selected_ids = set(references)
    selected_modes = set(modes)
    summary_records: list[dict[str, Any]] = []
    completed: set[tuple[str, str]] = set()
    offset = 0
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.endswith(b"\n"):
                return summary_records, completed, (offset, raw_line)
            offset += len(raw_line)
            if not raw_line.strip():
                raise ValueError(f"Cannot resume: blank results line {line_number}")
            try:
                record = json.loads(
                    raw_line,
                    parse_constant=_reject_nonfinite_json,
                    parse_float=parse_finite_float,
                )
            except (ValueError, OverflowError, UnicodeError) as exc:
                raise ValueError(
                    f"Cannot resume: invalid results JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Cannot resume: results line {line_number} is not an object")
            case_id = record.get("case_id")
            mode = record.get("mode")
            if (
                not isinstance(case_id, str)
                or not isinstance(mode, str)
                or case_id not in selected_ids
                or mode not in selected_modes
            ):
                raise ValueError(
                    f"Cannot resume: unexpected case/mode at results line {line_number}"
                )
            pair = (case_id, mode)
            if pair in completed:
                raise ValueError(
                    f"Cannot resume: duplicate case/mode at results line {line_number}"
                )
            if (
                len(summary_records) >= len(expected_order)
                or pair != expected_order[len(summary_records)]
            ):
                raise ValueError(
                    f"Cannot resume: results line {line_number} violates execution order"
                )
            try:
                summary_records.append(
                    _summary_record(record, spec, cases[case_id], references[case_id])
                )
            except ValueError as exc:
                raise ValueError(f"Cannot resume: results line {line_number}: {exc}") from exc
            completed.add(pair)
    return summary_records, completed, None


def _quarantine_trailing_result(
    path: Path, trailing: tuple[int, bytes]
) -> dict[str, Any]:
    """Save an unterminated final row before removing it from the ledger."""

    offset, fragment = trailing
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix="results-tail-",
        suffix=".bin",
        delete=False,
    ) as backup:
        backup.write(fragment)
        backup.flush()
        os.fsync(backup.fileno())
        backup_name = Path(backup.name).name
    _sync_directory(path.parent)
    with path.open("r+b") as ledger:
        ledger.seek(offset)
        if ledger.read() != fragment:
            raise ValueError("Cannot resume: results.jsonl changed during tail recovery")
        ledger.truncate(offset)
        ledger.flush()
        os.fsync(ledger.fileno())
    return {
        "file": backup_name,
        "sha256": hashlib.sha256(fragment).hexdigest(),
        "bytes": len(fragment),
    }


def _zero_stage() -> dict[str, Any]:
    return {
        "elapsed_seconds": 0.0,
        "cost_usd": 0.0,
        "recorded_cost_usd": 0.0,
        "cost_complete": True,
    }


def _planner_stage(planner: JudgePlanGenerator, elapsed: float) -> dict[str, Any]:
    usage = planner._usage
    input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
    output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
    estimate = float(planner.cost.total_price) if usage is not None else 0.0
    if not math.isfinite(estimate) or estimate < 0:
        estimate = 0.0
    # A Meta call consumed a prompt even when the provider reports zero tokens.
    # The chat price catalog also returns zero for unknown models.
    complete = usage is not None and estimate > 0
    return {
        "elapsed_seconds": elapsed,
        "cost_usd": estimate if complete else None,
        "recorded_cost_usd": estimate,
        "cost_complete": complete,
        "cost_source": "catalog_estimate",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _error_result(case_id: str, error: str, elapsed: float = 0.0) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "error",
        "answers": {},
        "findings": [],
        "calls": [],
        "elapsed_seconds": elapsed,
        "recorded_cost_usd": 0.0,
        "cost_usd": None,
        "cost_complete": False,
        "error": error,
    }


async def _run_one(
    case: EvaluationCase,
    reference: ReferenceCase,
    spec: EvaluationSpec,
    mode: str,
    evaluator: JevEvaluator,
    planner: JudgePlanGenerator | None,
    fixed_plan: JudgePlan | None,
    specialist_count: int | None,
) -> dict[str, Any]:
    started = perf_counter()
    plan: JudgePlan | None = None
    planner_stage = _zero_stage()
    judge_stage = _zero_stage()
    result: dict[str, Any]

    if mode == "fixed":
        plan = fixed_plan
    elif mode == "adaptive":
        if planner is None:
            raise RuntimeError("Adaptive mode requires a plan generator")
        planner._usage = None
        planning_started = perf_counter()
        try:
            plan = await planner.create_plan(
                case, spec, specialist_count=specialist_count
            )
        except Exception as exc:  # Preserve this case and mode in the denominator.
            planner_stage = _planner_stage(planner, perf_counter() - planning_started)
            result = _error_result(case.case_id, f"plan: {type(exc).__name__}: {exc}")
        else:
            planner_stage = _planner_stage(planner, perf_counter() - planning_started)
            if plan is None:
                result = _error_result(case.case_id, "plan: generator returned no plan")

    if mode != "adaptive" or plan is not None:
        judging_started = perf_counter()
        try:
            result = await evaluator.evaluate(case, spec, plan)
        except Exception as exc:  # The ledger still gets a failed paired attempt.
            result = _error_result(
                case.case_id,
                f"evaluate: {type(exc).__name__}: {exc}",
                elapsed=perf_counter() - judging_started,
            )
        judge_stage = {
            "elapsed_seconds": float(
                result.get("elapsed_seconds", perf_counter() - judging_started)
            ),
            "cost_usd": result.get("cost_usd"),
            "recorded_cost_usd": float(result.get("recorded_cost_usd", 0.0)),
            "cost_complete": bool(result.get("cost_complete", False)),
            "cost_source": "provider_receipts",
        }

    try:
        scores = score_case(spec, case, reference, result)
    except Exception as exc:
        # Preserve calls and costs if scoring finds a malformed evaluator result.
        result = {
            **result,
            "case_id": case.case_id,
            "status": "error",
            "answers": {},
            "error": f"score: {type(exc).__name__}: {exc}",
        }
        scores = score_case(spec, case, reference, result)

    status = str(result.get("status", "error"))
    projection_failures = [metric_id for metric_id, score in scores.items() if not score["valid"]]
    if projection_failures and status == "ok":
        status = "partial" if len(projection_failures) < len(scores) else "error"
    error = result.get("error")
    if projection_failures:
        detail = f"invalid target projections: {', '.join(projection_failures)}"
        error = f"{error}; {detail}" if error else detail

    total_elapsed = perf_counter() - started
    recorded_cost = planner_stage["recorded_cost_usd"] + judge_stage["recorded_cost_usd"]
    cost_complete = planner_stage["cost_complete"] and judge_stage["cost_complete"]
    total_cost = (
        planner_stage["cost_usd"] + judge_stage["cost_usd"] if cost_complete else None
    )
    return {
        "case_id": case.case_id,
        "mode": mode,
        "status": status,
        "evaluation_status": result.get("status"),
        "error": error,
        "reference": reference.labels,
        "answers": result.get("answers", {}),
        "findings": result.get("findings", []),
        "calls": result.get("calls", []),
        "plan": plan.model_dump(mode="json") if plan is not None else None,
        "scores": scores,
        "stages": {
            "planner": planner_stage,
            "judge": judge_stage,
            "total": {
                "elapsed_seconds": total_elapsed,
                "cost_usd": total_cost,
                "recorded_cost_usd": recorded_cost,
                "cost_complete": cost_complete,
            },
        },
    }


async def _run_locked(
    args: argparse.Namespace,
    prepared: tuple[
        EvaluationSpec,
        list[EvaluationCase],
        dict[str, ReferenceCase],
        list[str],
        JudgePlan | None,
    ] | None = None,
) -> dict[str, Any]:
    spec, cases, references, modes, fixed_plan = prepared or _preflight(args)
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --validate-only is used")
    case_ids = [case.case_id for case in cases]
    cases_by_id = {case.case_id: case for case in cases}
    ordered_pairs = _ordered_pairs(cases, modes)
    expected_manifest: dict[str, Any] = {
        "state": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": spec.benchmark_id,
        "spec_version": spec.version,
        "primary_metric": args.primary_metric,
        "case_ids": case_ids,
        "case_count": len(case_ids),
        "modes": modes,
        "inputs": _input_manifest(args),
        "source_sha256": _source_hashes(),
        "models": {
            "jev_requested": args.jev_model,
            "meta_requested": args.meta_model if "adaptive" in modes else None,
        },
        "planner_prompt_version": PROMPT_VERSION if "adaptive" in modes else None,
        "runtime": _resolved_runtime(args, modes),
        "routes": _route_manifest(modes),
        "dependency_versions": _dependency_versions(),
        "execution_order": MODE_ORDER_POLICY,
        "inflight_pair": None,
        "untracked_cost_modes": [],
        "quarantined_tails": [],
    }
    manifest_path = args.output_dir / "manifest.json"
    result_path = args.output_dir / "results.jsonl"
    summary_path = args.output_dir / "summary.json"
    if args.resume:
        if not manifest_path.is_file():
            raise ValueError("Cannot resume: output directory has no manifest.json")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing_manifest, dict):
            raise ValueError("Cannot resume: manifest.json is not an object")
        _validate_resume_manifest(existing_manifest, expected_manifest)
        state = existing_manifest.get("state")
        if not isinstance(state, str) or state not in {
            "running", "interrupted", "complete"
        }:
            raise ValueError("Cannot resume: invalid manifest state")
        resume_count = existing_manifest.get("resume_count", 0)
        if type(resume_count) is not int or resume_count < 0:
            raise ValueError("Cannot resume: invalid resume_count")
        attempts_written = existing_manifest.get("attempts_written")
        if type(attempts_written) is not int or attempts_written < 0:
            raise ValueError("Cannot resume: invalid manifest attempts_written")
        quarantined_tails = existing_manifest.get("quarantined_tails", [])
        if not isinstance(quarantined_tails, list):
            raise ValueError("Cannot resume: invalid quarantined_tails")
        prior_untracked = existing_manifest.get("untracked_cost_modes", [])
        if (
            not isinstance(prior_untracked, list)
            or any(not isinstance(mode, str) or mode not in modes for mode in prior_untracked)
        ):
            raise ValueError("Cannot resume: invalid untracked_cost_modes")
        untracked_cost_modes = set(prior_untracked)
        inflight = existing_manifest.get("inflight_pair")
        inflight_pair: tuple[str, str] | None = None
        if inflight is not None:
            if (
                not isinstance(inflight, dict)
                or not isinstance(inflight.get("case_id"), str)
                or inflight["case_id"] not in references
                or not isinstance(inflight.get("mode"), str)
                or inflight["mode"] not in modes
            ):
                raise ValueError("Cannot resume: invalid inflight_pair")
            inflight_pair = (inflight["case_id"], inflight["mode"])
        summary_records, completed, trailing = _load_existing_records(
            result_path, spec, cases_by_id, references, modes, ordered_pairs
        )
        if trailing is not None:
            if (
                state == "complete"
                or inflight_pair is None
                or len(summary_records) != attempts_written
                or inflight_pair in completed
                or len(summary_records) >= len(ordered_pairs)
                or ordered_pairs[len(summary_records)] != inflight_pair
            ):
                raise ValueError("Cannot resume: unterminated results tail is not the inflight pair")
            quarantined_tails.append(_quarantine_trailing_result(result_path, trailing))
            existing_manifest["quarantined_tails"] = quarantined_tails
            # Tail recovery mutates the ledger. Persist its backup reference
            # before credential checks or planner construction can fail.
            _write_json(manifest_path, existing_manifest)
        actual_written = len(summary_records)
        last_pair = (
            (summary_records[-1]["case_id"], summary_records[-1]["mode"])
            if summary_records
            else None
        )
        if actual_written < attempts_written:
            raise ValueError("Cannot resume: results.jsonl is missing recorded attempts")
        if actual_written > attempts_written:
            if actual_written != attempts_written + 1 or last_pair != inflight_pair:
                raise ValueError("Cannot resume: results.jsonl exceeds manifest progress")
        elif inflight_pair in completed and last_pair != inflight_pair:
            raise ValueError("Cannot resume: completed inflight_pair conflicts with manifest progress")
        if state == "complete" and actual_written != len(ordered_pairs):
            raise ValueError("Cannot resume: complete manifest has missing result pairs")
        if state == "complete" and inflight_pair is not None:
            raise ValueError("Cannot resume: complete manifest has an inflight pair")
        if inflight_pair is not None:
            if inflight_pair not in completed:
                if (
                    actual_written >= len(ordered_pairs)
                    or ordered_pairs[actual_written] != inflight_pair
                ):
                    raise ValueError("Cannot resume: inflight_pair is not the next pair")
                # The provider may have charged a call before a durable result row.
                untracked_cost_modes.add(inflight["mode"])
        existing_manifest["untracked_cost_modes"] = sorted(untracked_cost_modes)
        existing_manifest["inflight_pair"] = None
        manifest = existing_manifest
    else:
        summary_records = []
        completed = set()
        manifest = expected_manifest

    expected_pairs = set(ordered_pairs)
    pending = expected_pairs - completed
    if pending:
        decision_connection()  # Check credentials before writing run state.
        planner = (
            JudgePlanGenerator(model=args.meta_model, temperature=args.meta_temperature)
            if any(mode == "adaptive" for _, mode in pending)
            else None
        )
    else:
        planner = None

    if args.resume:
        manifest["state"] = "running"
        manifest["resume_count"] = resume_count + 1
        manifest["last_resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest.pop("finished_at_utc", None)
    manifest["attempts_written"] = len(summary_records)
    _write_json(manifest_path, manifest)

    # The full receipts go to JSONL. Keep only scores and stage accounting in
    # memory for summary, so long traces do not multiply RAM use unnecessarily.
    try:
        if pending:
            async with model_http_client() as client:
                evaluator = JevEvaluator(
                    client,
                    model=args.jev_model,
                    max_attempts=args.max_attempts,
                    max_concurrent_requests=args.max_concurrent_requests,
                    batch_size=args.batch_size,
                )
                with result_path.open("a" if args.resume else "x", encoding="utf-8") as output:
                    for case_id, mode in ordered_pairs:
                        case = cases_by_id[case_id]
                        reference = references[case.case_id]
                        if (case_id, mode) in completed:
                            continue
                        manifest["inflight_pair"] = {"case_id": case_id, "mode": mode}
                        _write_json(manifest_path, manifest)
                        record = await _run_one(
                            case,
                            reference,
                            spec,
                            mode,
                            evaluator,
                            planner,
                            fixed_plan,
                            args.specialist_count,
                        )
                        summary_record = _summary_record(record, spec, case, reference)
                        output.write(
                            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                        )
                        output.flush()
                        os.fsync(output.fileno())
                        summary_records.append(summary_record)
                        manifest["attempts_written"] = len(summary_records)
                        manifest["inflight_pair"] = None
                        _write_json(manifest_path, manifest)
        summary = summarize_runs(
            spec,
            references,
            summary_records,
            modes,
            untracked_modes=manifest["untracked_cost_modes"],
        )
        summary["primary_metric"] = args.primary_metric
        _write_json(summary_path, summary)
    except BaseException:
        manifest["state"] = "interrupted"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["attempts_written"] = len(summary_records)
        _write_json(manifest_path, manifest)
        raise

    manifest["state"] = "complete"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["attempts_written"] = len(summary_records)
    _write_json(manifest_path, manifest)
    return summary


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --validate-only is used")
    if args.resume:
        if not args.output_dir.is_dir():
            raise ValueError("Cannot resume: output directory does not exist")
        with _exclusive_run_lock(args.output_dir):
            return await _run_locked(args)

    prepared = _preflight(args)
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    decision_connection()  # Check credentials before creating an output directory.
    if "adaptive" in prepared[3]:
        chat_api_key()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with _exclusive_run_lock(args.output_dir):
        return await _run_locked(args, prepared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="Normalized cases JSONL")
    parser.add_argument("--references", type=Path, required=True, help="Held-out references JSONL")
    parser.add_argument("--spec", type=Path, required=True, help="Evaluation spec JSON")
    parser.add_argument(
        "--primary-metric",
        required=True,
        help="Predeclared metric ID from the spec, fixed in the run manifest",
    )
    parser.add_argument("--output-dir", type=Path, help="Experiment directory")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["direct", "adaptive"])
    parser.add_argument("--fixed-plan", type=Path, help="JudgePlan JSON for fixed mode")
    parser.add_argument("--jev-model", default=JEV_MODEL)
    parser.add_argument(
        "--meta-model", default=os.getenv("DEFAULT_META_MODEL", "anthropic/claude-sonnet-4")
    )
    parser.add_argument("--meta-temperature", type=float, default=0.0)
    parser.add_argument(
        "--specialist-count",
        type=int,
        help="Exact specialist count for adaptive; fixed/adaptive defaults to fixed plan count",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=4,
        help="Maximum simultaneous Decisions requests within one case",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Evaluate the first N normalized cases")
    parser.add_argument(
        "--case-ids-file",
        type=Path,
        help="Text file with one selected case ID per line, in execution order",
    )
    parser.add_argument(
        "--allow-unmanifested-input",
        action="store_true",
        help="Accept manually prepared cases and references without a converter manifest",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check cases, references, spec, and fixed plan without API credentials or output",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Continue an existing experiment ledger"
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_attempts < 1 or args.max_concurrent_requests < 1 or args.batch_size < 1:
        parser.error("--max-attempts, --max-concurrent-requests, and --batch-size must be positive")
    if args.specialist_count is not None and args.specialist_count < 1:
        parser.error("--specialist-count must be positive")
    if not 0.0 <= args.meta_temperature <= 1.0:
        parser.error("--meta-temperature must be between 0 and 1")
    if args.validate_only and args.resume:
        parser.error("--validate-only and --resume cannot be combined")
    if not args.validate_only and args.output_dir is None:
        parser.error("--output-dir is required unless --validate-only is used")
    if args.validate_only:
        spec, cases, _, modes, fixed_plan = _preflight(args)
        print(
            json.dumps(
                {
                    "benchmark_id": spec.benchmark_id,
                    "spec_version": spec.version,
                    "primary_metric": args.primary_metric,
                    "case_count": len(cases),
                    "case_ids": [case.case_id for case in cases],
                    "modes": modes,
                    "specialist_count": args.specialist_count,
                    "fixed_plan_specialists": (
                        len(fixed_plan.specialists) if fixed_plan is not None else None
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    summary = asyncio.run(run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if any(
        arm["status"]["partial"] or arm["status"]["error"]
        for arm in summary["modes"].values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
