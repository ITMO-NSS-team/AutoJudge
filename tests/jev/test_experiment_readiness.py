"""Offline regression checks for Jev experiment selection and recovery."""

import argparse
import hashlib
import json
from pathlib import Path
from runpy import run_path
from typing import Any

import pytest

from autojudge.jev.contracts import (
    EvaluationCase,
    EvaluationSpec,
    EvidenceEvent,
    MetricSpec,
    Option,
    ReferenceCase,
    TargetSpec,
)
from autojudge.jev.scoring import score_case, summarize_runs


def _runner() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "examples/jev/run_experiment.py"
    return run_path(str(path))


def _spec() -> EvaluationSpec:
    return EvaluationSpec(
        benchmark_id="offline-regression",
        taxonomy="",
        targets=[
            TargetSpec(
                id="decision",
                question="Which answer is supported?",
                source="options",
                options=[
                    Option(id="yes", description="Yes"),
                    Option(id="no", description="No"),
                ],
            )
        ],
        metrics=[MetricSpec(id="accuracy", target="decision", label="answer")],
    )


def _case(case_id: str) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        task="Choose an answer",
        events=[EvidenceEvent(id="e1", index=1, content="Evidence")],
    )


def _reference(case_id: str) -> ReferenceCase:
    return ReferenceCase(case_id=case_id, labels={"answer": "yes"})


def _write_inputs(
    tmp_path: Path, case_ids: tuple[str, ...] = ("one", "two", "three")
) -> tuple[Path, Path, Path]:
    cases_path = tmp_path / "cases.jsonl"
    references_path = tmp_path / "references.jsonl"
    spec_path = tmp_path / "spec.json"
    cases_path.write_text(
        "".join(_case(case_id).model_dump_json() + "\n" for case_id in case_ids),
        encoding="utf-8",
    )
    references_path.write_text(
        "".join(_reference(case_id).model_dump_json() + "\n" for case_id in case_ids),
        encoding="utf-8",
    )
    spec_path.write_text(_spec().model_dump_json(), encoding="utf-8")
    return cases_path, references_path, spec_path


def _record(
    mode: str,
    case_id: str,
    value: float,
    *,
    cost: float,
    elapsed: float,
    valid: bool = True,
) -> dict[str, Any]:
    stages = {
        name: {
            "elapsed_seconds": elapsed if name != "planner" else 0.0,
            "cost_usd": cost if name != "planner" else 0.0,
            "recorded_cost_usd": cost if name != "planner" else 0.0,
            "cost_complete": True,
        }
        for name in ("planner", "judge", "total")
    }
    return {
        "mode": mode,
        "case_id": case_id,
        "status": "ok" if valid else "error",
        "scores": {"accuracy": {"valid": valid, "value": value}},
        "jev_calls": 1,
        "http_attempts": 1,
        "retries": 0,
        "stages": stages,
    }


def _ledger_record(
    spec: EvaluationSpec,
    case: EvaluationCase,
    reference: ReferenceCase,
    mode: str,
) -> dict[str, Any]:
    answers = {
        "decision": {
            "status": "ok",
            "selected_ids": ["yes"],
            "selected": [{"id": "yes"}],
        }
    }
    return {
        **_record(mode, case.case_id, 1.0, cost=0.1, elapsed=2.0),
        "reference": reference.labels,
        "answers": answers,
        "calls": [],
        "scores": score_case(
            spec, case, reference, {"case_id": case.case_id, "answers": answers}
        ),
    }


def test_case_ids_file_preserves_declared_order_and_selected_references(
    tmp_path: Path,
) -> None:
    cases_path, references_path, spec_path = _write_inputs(tmp_path)
    selection = tmp_path / "selected.txt"
    selection.write_text("\ufeffthree\none\n", encoding="utf-8")

    _, cases, references = _runner()["_load_inputs"](
        cases_path,
        references_path,
        spec_path,
        None,
        case_ids_file=selection,
        allow_unmanifested=True,
    )

    assert [case.case_id for case in cases] == ["three", "one"]
    assert list(references) == ["three", "one"]

    selection.write_text("one\none\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate case ID"):
        _runner()["_load_inputs"](
            cases_path,
            references_path,
            spec_path,
            None,
            case_ids_file=selection,
            allow_unmanifested=True,
        )


def test_preflight_rejects_unknown_primary_metric_and_conflicting_selection(
    tmp_path: Path,
) -> None:
    cases_path, references_path, spec_path = _write_inputs(tmp_path)
    selection = tmp_path / "selected.txt"
    selection.write_text("two\n", encoding="utf-8")
    args = argparse.Namespace(
        cases=cases_path,
        references=references_path,
        spec=spec_path,
        limit=None,
        case_ids_file=selection,
        allow_unmanifested_input=True,
        primary_metric="accuracy",
        modes=["direct"],
        fixed_plan=None,
        specialist_count=None,
    )
    preflight = _runner()["_preflight"]

    assert [case.case_id for case in preflight(args)[1]] == ["two"]
    args.primary_metric = "missing"
    with pytest.raises(ValueError, match="--primary-metric"):
        preflight(args)
    args.primary_metric = "accuracy"
    args.limit = 1
    with pytest.raises(ValueError, match="--case-ids-file and --limit"):
        preflight(args)


def test_adaptive_is_compared_to_fixed_on_the_same_selected_cases() -> None:
    spec = _spec()
    references = {case_id: _reference(case_id) for case_id in ("one", "two")}
    records = [
        _record("fixed", "one", 0.0, cost=0.1, elapsed=2.0),
        _record("adaptive", "one", 1.0, cost=0.2, elapsed=3.0),
        _record("fixed", "two", 0.0, cost=0.1, elapsed=2.0),
        _record("adaptive", "two", 0.0, cost=0.2, elapsed=3.0, valid=False),
    ]

    summary = summarize_runs(spec, references, records, ["fixed", "adaptive"])
    delta = summary["delta_vs_fixed"]["adaptive"]

    assert summary["cases"] == 2
    assert "delta_vs_direct" not in summary
    assert summary["modes"]["adaptive"]["completion_rate"] == pytest.approx(0.5)
    assert summary["modes"]["adaptive"]["metrics"]["accuracy"][
        "coverage"
    ] == pytest.approx(0.5)
    assert delta["metrics"]["accuracy"] == pytest.approx(0.5)
    assert delta["exact_metric_case_comparisons"]["accuracy"] == {
        "wins": 1,
        "losses": 0,
        "ties": 1,
    }
    assert delta["total_cost_usd"] == pytest.approx(0.2)
    assert delta["quality_gain_per_extra_usd"]["accuracy"] == pytest.approx(2.5)


def test_resumed_untracked_mode_retains_only_observed_cost_and_latency() -> None:
    spec = _spec()
    references = {"one": _reference("one")}
    records = [
        _record("fixed", "one", 0.0, cost=0.1, elapsed=2.0),
        _record("adaptive", "one", 1.0, cost=0.2, elapsed=3.0),
    ]

    # Resume marks a mode untracked if an inflight request had no durable result row.
    summary = summarize_runs(
        spec, references, records, ["fixed", "adaptive"], untracked_modes=["adaptive"]
    )
    total = summary["modes"]["adaptive"]["stages"]["total"]
    delta = summary["delta_vs_fixed"]["adaptive"]

    assert summary["untracked_cost_modes"] == ["adaptive"]
    assert summary["untracked_latency_modes"] == ["adaptive"]
    assert total["recorded_cost_usd"] == pytest.approx(0.2)
    assert total["recorded_total_seconds"] == pytest.approx(3.0)
    assert total["cost_complete_cases"] == 1
    assert total["latency_complete"] is False
    assert all(
        total[key] is None
        for key in (
            "total_cost_usd",
            "total_seconds",
            "mean_seconds",
            "p50_seconds",
            "p95_seconds",
        )
    )
    assert delta["total_cost_usd"] is None
    assert delta["mean_total_seconds"] is None
    assert delta["quality_gain_per_extra_usd"]["accuracy"] is None


def test_unterminated_result_tail_is_preserved_before_resume_truncation(
    tmp_path: Path,
) -> None:
    runner = _runner()
    spec = _spec()
    case = _case("one")
    reference = _reference("one")
    valid_row = _ledger_record(spec, case, reference, "fixed")
    prefix = (json.dumps(valid_row) + "\n").encode("utf-8")
    fragment = b'{"case_id":"one","mode":"adaptive"'
    results = tmp_path / "results.jsonl"
    results.write_bytes(prefix + fragment)

    rows, completed, trailing = runner["_load_existing_records"](
        results,
        spec,
        {"one": case},
        {"one": reference},
        ["fixed", "adaptive"],
        [("one", "fixed"), ("one", "adaptive")],
    )

    assert len(rows) == 1
    assert completed == {("one", "fixed")}
    assert trailing is not None
    assert trailing == (len(prefix), fragment)
    backup = runner["_quarantine_trailing_result"](results, trailing)
    assert results.read_bytes() == prefix
    assert (tmp_path / backup["file"]).read_bytes() == fragment
    assert backup["sha256"] == hashlib.sha256(fragment).hexdigest()
    assert backup["bytes"] == len(fragment)


@pytest.mark.asyncio
async def test_resume_accepts_durable_row_with_stale_inflight_marker(
    tmp_path: Path,
) -> None:
    runner = _runner()
    cases_path, references_path, spec_path = _write_inputs(tmp_path, ("one",))
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    spec = _spec()
    case = _case("one")
    reference = _reference("one")
    args = argparse.Namespace(
        output_dir=output_dir,
        resume=True,
        cases=cases_path,
        references=references_path,
        spec=spec_path,
        case_ids_file=None,
        fixed_plan=None,
        primary_metric="accuracy",
        jev_model="test-jev",
        meta_model="unused",
        meta_temperature=0.0,
        specialist_count=None,
        max_attempts=3,
        max_concurrent_requests=4,
        batch_size=16,
    )
    modes = ["direct"]
    manifest = {
        "state": "interrupted",
        "benchmark_id": spec.benchmark_id,
        "spec_version": spec.version,
        "primary_metric": "accuracy",
        "case_ids": ["one"],
        "case_count": 1,
        "modes": modes,
        "inputs": runner["_input_manifest"](args),
        "source_sha256": runner["_source_hashes"](),
        "models": {"jev_requested": args.jev_model, "meta_requested": None},
        "planner_prompt_version": None,
        "runtime": runner["_resolved_runtime"](args, modes),
        "routes": runner["_route_manifest"](modes),
        "dependency_versions": runner["_dependency_versions"](),
        "execution_order": runner["MODE_ORDER_POLICY"],
        "attempts_written": 1,
        "inflight_pair": {"case_id": "one", "mode": "direct"},
        "untracked_cost_modes": [],
        "quarantined_tails": [],
        "resume_count": 0,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output_dir / "results.jsonl").write_text(
        json.dumps(_ledger_record(spec, case, reference, "direct")) + "\n",
        encoding="utf-8",
    )

    summary = await runner["_run_locked"](
        args, (spec, [case], {"one": reference}, modes, None)
    )

    saved = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saved["state"] == "complete"
    assert saved["attempts_written"] == 1
    assert saved["inflight_pair"] is None
    assert saved["resume_count"] == 1
    assert saved["untracked_cost_modes"] == []
    assert summary["untracked_cost_modes"] == []
    assert (output_dir / "results.jsonl").read_text(encoding="utf-8").count("\n") == 1
    total = summary["modes"]["direct"]["stages"]["total"]
    assert total["latency_complete"] is True
    assert total["total_cost_usd"] == pytest.approx(0.1)
