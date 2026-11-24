"""Contract and scoring checks for normalized Jev benchmark cases."""

from typing import Any

import pytest
from pydantic import ValidationError

from autojudge.jev.contracts import (
    EvaluationCase,
    EvaluationSpec,
    EvidenceEvent,
    MetricSpec,
    Option,
    ReferenceCase,
    TargetSpec,
)
from autojudge.jev.preflight import validate_case_against_spec
from autojudge.jev.scoring import score_case, summarize_runs


def _binary_spec() -> EvaluationSpec:
    return EvaluationSpec(
        benchmark_id="binary",
        taxonomy="",
        targets=[
            TargetSpec(
                id="decision",
                question="Was the condition met?",
                source="options",
                options=[
                    Option(id="Yes", description="Condition met"),
                    Option(id="No", description="Condition not met"),
                ],
            )
        ],
        metrics=[
            MetricSpec(
                id="f1",
                target="decision",
                kind="binary_f1",
                label="positive",
                positive_id="Yes",
            ),
            MetricSpec(
                id="precision",
                target="decision",
                kind="binary_precision",
                label="positive",
                positive_id="Yes",
            ),
        ],
    )


def _result(case_id: str, target_id: str, selected_id: str | None) -> dict[str, Any]:
    answer = (
        {"status": "error", "selected_ids": [], "selected": []}
        if selected_id is None
        else {
            "status": "ok",
            "selected_ids": [selected_id],
            "selected": [{"id": selected_id}],
        }
    )
    return {"case_id": case_id, "answers": {target_id: answer}}


def _case(case_id: str) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        task="Evaluate the case",
        events=[EvidenceEvent(id="e1", index=1, content="Evidence")],
    )


def _record(
    spec: EvaluationSpec,
    reference: ReferenceCase,
    selected_id: str | None,
    target_id: str = "decision",
) -> dict[str, Any]:
    stages = {
        name: {
            "elapsed_seconds": 0.0,
            "cost_usd": 0.0,
            "recorded_cost_usd": 0.0,
            "cost_complete": True,
        }
        for name in ("planner", "judge", "total")
    }
    return {
        "mode": "direct",
        "case_id": reference.case_id,
        "status": "ok" if selected_id is not None else "error",
        "scores": score_case(
            spec, _case(reference.case_id), reference,
            _result(reference.case_id, target_id, selected_id),
        ),
        "jev_calls": 0,
        "http_attempts": 0,
        "retries": 0,
        "stages": stages,
    }


def test_contract_rejects_unknown_binary_positive_option() -> None:
    with pytest.raises(ValidationError, match="unknown positive_id"):
        EvaluationSpec(
            benchmark_id="invalid",
            taxonomy="",
            targets=[
                TargetSpec(
                    id="decision",
                    question="Choose one",
                    source="options",
                    options=[Option(id="Yes", description="Yes")],
                )
            ],
            metrics=[
                MetricSpec(
                    id="f1",
                    target="decision",
                    kind="binary_f1",
                    label="positive",
                    positive_id="Missing",
                )
            ],
        )


def test_preflight_requires_jointly_reachable_agent_and_step() -> None:
    case = EvaluationCase(
        case_id="trajectory",
        task="Find the first mistake",
        events=[
            EvidenceEvent(id="e1", index=1, actor="Alice", content="First"),
            EvidenceEvent(id="e2", index=2, actor="Bob", content="Second"),
        ],
    )
    spec = EvaluationSpec(
        benchmark_id="attribution",
        taxonomy="",
        targets=[TargetSpec(id="origin", question="Which event?", source="events")],
        metrics=[
            MetricSpec(id="agent", target="origin", projection="actor", label="agent"),
            MetricSpec(id="step", target="origin", projection="index", label="step"),
        ],
    )

    with pytest.raises(ValueError, match="no selection satisfies metrics"):
        validate_case_against_spec(
            case,
            spec,
            ReferenceCase(case_id="trajectory", labels={"agent": "Alice", "step": 2}),
        )

    validate_case_against_spec(
        case,
        spec,
        ReferenceCase(case_id="trajectory", labels={"agent": "Bob", "step": 2}),
    )


def test_preflight_many_metrics_need_one_compatible_candidate_subset() -> None:
    case = EvaluationCase(
        case_id="many",
        task="Select events",
        events=[
            EvidenceEvent(id="e1", index=1, actor="Alice", content="First"),
            EvidenceEvent(id="e2", index=2, actor="Bob", content="Second"),
        ],
    )
    spec = EvaluationSpec(
        benchmark_id="many",
        taxonomy="",
        targets=[
            TargetSpec(id="events", question="Which events?", source="events", cardinality="many")
        ],
        metrics=[
            MetricSpec(id="actors", target="events", projection="actor", label="actors"),
            MetricSpec(
                id="steps", target="events", kind="set_f1", projection="index", label="steps"
            ),
        ],
    )

    with pytest.raises(ValueError, match="no selection satisfies metrics"):
        validate_case_against_spec(
            case,
            spec,
            ReferenceCase(case_id="many", labels={"actors": ["Alice"], "steps": [2]}),
        )

    validate_case_against_spec(
        case,
        spec,
        ReferenceCase(case_id="many", labels={"actors": ["Alice", "Bob"], "steps": [1, 2]}),
    )


def test_binary_f1_and_precision_count_valid_and_invalid_answers() -> None:
    spec = _binary_spec()
    references = {
        case_id: ReferenceCase(case_id=case_id, labels={"positive": positive})
        for case_id, positive in (
            ("true_positive", True),
            ("true_negative", False),
            ("failed_positive", True),
            ("failed_negative", False),
        )
    }
    selections = {
        "true_positive": "Yes",
        "true_negative": "No",
        "failed_positive": None,
        "failed_negative": None,
    }
    records = [
        _record(spec, reference, selections[case_id]) for case_id, reference in references.items()
    ]

    metrics = summarize_runs(spec, references, records, ["direct"])["modes"]["direct"][
        "metrics"
    ]
    for metric_id in ("f1", "precision"):
        result = metrics[metric_id]
        assert result["value"] == pytest.approx(0.5)
        assert result["denominator"] == 4
        assert result["invalid_answers"] == 2
        assert (result["tp"], result["tn"], result["fp"], result["fn"]) == (1, 1, 1, 1)


def test_mae_overflow_uses_failure_penalty() -> None:
    spec = EvaluationSpec(
        benchmark_id="numeric",
        taxonomy="",
        targets=[
            TargetSpec(
                id="value",
                question="Choose a number",
                source="options",
                options=[
                    Option(id="1e308", description="Positive large number"),
                    Option(id="-1e308", description="Negative large number"),
                    Option(id="0", description="Zero"),
                ],
            )
        ],
        metrics=[
            MetricSpec(
                id="mae",
                target="value",
                kind="mae",
                label="expected",
                failure_penalty=7.0,
            )
        ],
    )
    reference = ReferenceCase(case_id="overflow", labels={"expected": "-1e308"})

    score = score_case(spec, _case("overflow"), reference, _result("overflow", "value", "1e308"))["mae"]
    assert score["valid"] is False
    assert score["value"] == 7.0
    assert score["error"] == "Absolute error is not finite"

    references = {
        case_id: ReferenceCase(case_id=case_id, labels={"expected": 0})
        for case_id in ("large_1", "large_2")
    }
    records = [
        _record(spec, reference, "1e308", target_id="value") for reference in references.values()
    ]
    mean_mae = summarize_runs(spec, references, records, ["direct"])["modes"]["direct"][
        "metrics"
    ]["mae"]["value"]
    assert mean_mae == pytest.approx(1e308)


def test_unknown_static_option_id_is_invalid_not_a_negative_prediction() -> None:
    spec = _binary_spec()
    reference = ReferenceCase(case_id="unknown", labels={"positive": False})

    scores = score_case(spec, _case("unknown"), reference, _result("unknown", "decision", "Unknown"))
    for score in scores.values():
        assert score["valid"] is False
        assert score["value"] == 0.0
        assert score["prediction_positive"] is None
        assert "not declared" in score["error"]


def test_event_projection_rejects_changed_candidate_fields() -> None:
    case = EvaluationCase(
        case_id="event",
        task="Find the actor",
        events=[EvidenceEvent(id="e1", index=1, actor="Alice", content="Evidence")],
    )
    spec = EvaluationSpec(
        benchmark_id="actor",
        taxonomy="",
        targets=[TargetSpec(id="origin", question="Which event?", source="events")],
        metrics=[MetricSpec(id="actor", target="origin", projection="actor", label="actor")],
    )
    reference = ReferenceCase(case_id="event", labels={"actor": "Bob"})
    result = {
        "case_id": "event",
        "answers": {
            "origin": {
                "status": "ok",
                "selected_ids": ["e1"],
                "selected": [{"id": "e1", "actor": "Bob"}],
            }
        },
    }

    score = score_case(spec, case, reference, result)["actor"]
    assert score["valid"] is False
    assert score["value"] == 0.0
    assert "differs from the case" in score["error"]


def test_paired_comparisons_and_gain_per_extra_usd_use_metric_direction() -> None:
    spec = EvaluationSpec(
        benchmark_id="paired",
        taxonomy="",
        targets=[
            TargetSpec(
                id="choice",
                question="Which value?",
                source="options",
                options=[
                    Option(id="1", description="One"),
                    Option(id="2", description="Two"),
                ],
            )
        ],
        metrics=[
            MetricSpec(id="first", target="choice", label="first"),
            MetricSpec(id="second", target="choice", label="second"),
            MetricSpec(
                id="mae",
                target="choice",
                kind="mae",
                label="number",
                failure_penalty=10.0,
            ),
        ],
    )
    reference = ReferenceCase(
        case_id="paired_case", labels={"first": "1", "second": "2", "number": 2}
    )
    direct = _record(spec, reference, "1", target_id="choice")
    fixed = _record(spec, reference, "2", target_id="choice")
    fixed["mode"] = "fixed"
    for record, cost in ((direct, 1.0), (fixed, 2.0)):
        for stage_name in ("judge", "total"):
            record["stages"][stage_name]["cost_usd"] = cost
            record["stages"][stage_name]["recorded_cost_usd"] = cost

    summary = summarize_runs(
        spec, {reference.case_id: reference}, [direct, fixed], ["direct", "fixed"]
    )
    delta = summary["delta_vs_direct"]["fixed"]
    assert delta["exact_metric_case_comparisons"] == {
        "first": {"wins": 0, "losses": 1, "ties": 0},
        "second": {"wins": 1, "losses": 0, "ties": 0},
    }
    assert delta["total_cost_usd"] == pytest.approx(1.0)
    assert delta["quality_gain_per_extra_usd"]["first"] == pytest.approx(-1.0)
    assert delta["quality_gain_per_extra_usd"]["second"] == pytest.approx(1.0)
    assert delta["quality_gain_per_extra_usd"]["mae"] == pytest.approx(1.0)

    fixed["stages"]["judge"]["cost_usd"] = 1.0
    fixed["stages"]["total"]["cost_usd"] = 1.0
    fixed["stages"]["judge"]["recorded_cost_usd"] = 1.0
    fixed["stages"]["total"]["recorded_cost_usd"] = 1.0
    zero_cost_delta = summarize_runs(
        spec, {reference.case_id: reference}, [direct, fixed], ["direct", "fixed"]
    )["delta_vs_direct"]["fixed"]
    assert all(
        value is None for value in zero_cost_delta["quality_gain_per_extra_usd"].values()
    )
