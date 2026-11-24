"""Benchmark-neutral scoring of typed Jev answers against held-out references.

The evaluator never receives a :class:`ReferenceCase`. References enter only
here, after inference has produced a result. Every selected case contributes
to its metric, including missing answers and failed calls.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import EvaluationCase, EvaluationSpec, MetricSpec, ReferenceCase, TargetSpec


def _target_for(spec: EvaluationSpec, target_id: str) -> TargetSpec:
    return next(target for target in spec.targets if target.id == target_id)


def _reference_value(reference: ReferenceCase, metric: MetricSpec) -> Any:
    if metric.label not in reference.labels:
        raise ValueError(
            f"Reference {reference.case_id!r} has no label {metric.label!r} "
            f"for metric {metric.id!r}"
        )
    return reference.labels[metric.label]


def _finite_number(value: Any, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{description} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{description} must be a finite number")
    return number


def _hashable_set(value: Any, *, description: str) -> set[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be a JSON array")
    try:
        result = set(value)
    except TypeError as exc:
        raise ValueError(f"{description} must contain scalar values") from exc
    if len(result) != len(value):
        raise ValueError(f"{description} must not contain duplicates")
    return result


def _projected_set(value: Any) -> set[Any]:
    """Distinct projected labels; several selected events may share one actor."""

    if not isinstance(value, list):
        raise ValueError("Predicted values must be a JSON array")
    try:
        return set(value)
    except TypeError as exc:
        raise ValueError("Predicted values must contain scalar values") from exc


def _binary_reference(value: Any, metric: MetricSpec, reference: ReferenceCase) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        return value == metric.positive_id
    if value is None:
        return metric.positive_id == "none"
    raise ValueError(
        f"Reference {reference.case_id!r} label {metric.label!r} "
        "must be a boolean, 0/1, option ID, or null"
    )


def validate_reference(spec: EvaluationSpec, reference: ReferenceCase) -> None:
    """Reject malformed labels before spending money on inference."""

    for metric in spec.metrics:
        value = _reference_value(reference, metric)
        if metric.kind in {"binary_f1", "binary_precision"}:
            target = _target_for(spec, metric.target)
            if value is None and not target.allow_none:
                raise ValueError(
                    f"Reference {reference.case_id!r} label {metric.label!r} "
                    "uses no-match but the target does not allow it"
                )
            _binary_reference(value, metric, reference)
            if isinstance(value, str):
                valid_ids = {option.id for option in target.options}
                if target.allow_none:
                    valid_ids.add("none")
                if value not in valid_ids:
                    raise ValueError(
                        f"Reference {reference.case_id!r} label {metric.label!r} "
                        f"has unknown option ID {value!r}"
                    )
        elif metric.kind == "set_f1":
            _hashable_set(
                value,
                description=f"Reference {reference.case_id!r} label {metric.label!r}",
            )
        elif metric.kind == "mae":
            _finite_number(
                value,
                description=f"Reference {reference.case_id!r} label {metric.label!r}",
            )
        elif metric.kind == "exact" and _target_for(spec, metric.target).cardinality == "many":
            _hashable_set(
                value,
                description=f"Reference {reference.case_id!r} label {metric.label!r}",
            )


def _project(
    answer: Any, metric: MetricSpec, target: TargetSpec, case: EvaluationCase
) -> tuple[bool, Any, str | None]:
    if not isinstance(answer, Mapping):
        return False, None, "missing target answer"

    status = answer.get("status")
    if status == "none":
        if not target.allow_none:
            return False, None, "target does not allow no-match"
        if answer.get("selected") != [] or answer.get("selected_ids") != []:
            return False, None, "no-match answer contains selected candidates"
        return True, [] if target.cardinality == "many" else None, None
    if status != "ok":
        return False, None, f"target status is {status!r}"

    selected = answer.get("selected")
    selected_ids = answer.get("selected_ids")
    if not isinstance(selected, list) or not isinstance(selected_ids, list):
        return False, None, "selected and selected_ids must be arrays"
    if len(selected) != len(selected_ids):
        return False, None, "selected and selected_ids have different lengths"
    if not selected or (target.cardinality == "one" and len(selected) != 1):
        return False, None, "answer violates target cardinality"
    if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in selected_ids):
        return False, None, "selected IDs must be non-empty strings"
    if len(set(selected_ids)) != len(selected_ids):
        return False, None, "selected IDs contain duplicates"
    if any(
        not isinstance(item, Mapping) or item.get("id") != candidate_id
        for item, candidate_id in zip(selected, selected_ids)
    ):
        return False, None, "selected candidates do not match selected IDs"
    canonical: dict[str, dict[str, Any]]
    if target.source == "events":
        canonical = {
            event.id: {
                "actor": event.actor,
                "index": event.index,
                "source_step": event.source_step,
                "source_id": event.source_id,
            }
            for event in case.events
        }
        if any(candidate_id not in canonical for candidate_id in selected_ids):
            return False, None, "selected event ID is not present in the case"
    elif target.source == "actors":
        actor_ids = {event.actor for event in case.events if event.actor}
        canonical = {
            actor: {"actor": actor, "index": None, "source_step": None, "source_id": None}
            for actor in actor_ids
        }
        if any(candidate_id not in canonical for candidate_id in selected_ids):
            return False, None, "selected actor ID is not present in the case"
    else:
        canonical = {
            option.id: {"actor": None, "index": None, "source_step": None, "source_id": None}
            for option in target.options
        }
        if any(candidate_id not in canonical for candidate_id in selected_ids):
            return False, None, "selected option ID is not declared in the spec"
    for item, candidate_id in zip(selected, selected_ids):
        for field, expected in canonical[candidate_id].items():
            if field in item and (
                type(item[field]) is not type(expected) or item[field] != expected
            ):
                source = "spec" if target.source == "options" else "case"
                return False, None, f"selected candidate {field} differs from the {source}"

    if metric.projection == "id":
        values = selected_ids
    else:
        values = [canonical[candidate_id][metric.projection] for candidate_id in selected_ids]
    if any(value is None for value in values):
        return False, None, "selected value is null"
    if metric.projection == "actor" and any(
        not isinstance(value, str) or not value for value in values
    ):
        return False, None, "selected actor must be a non-empty string"
    if metric.projection in {"index", "source_step"}:
        minimum = 1 if metric.projection == "index" else 0
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < minimum
            for value in values
        ):
            return False, None, f"selected {metric.projection} must be an integer >= {minimum}"

    return True, values if target.cardinality == "many" else values[0], None


def _binary_prediction(predicted: Any, metric: MetricSpec) -> bool:
    if predicted is None:
        return metric.positive_id == "none"
    if isinstance(predicted, bool):
        return predicted
    return predicted == metric.positive_id


def score_case(
    spec: EvaluationSpec,
    case: EvaluationCase,
    reference: ReferenceCase,
    result: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Score typed answers, projecting candidate fields from the trusted case."""

    if case.case_id != reference.case_id or result.get("case_id") != case.case_id:
        raise ValueError("Evaluator, case, and reference case IDs differ")
    validate_reference(spec, reference)
    answers = result.get("answers")
    if not isinstance(answers, Mapping):
        answers = {}

    scores: dict[str, dict[str, Any]] = {}
    for metric in spec.metrics:
        target = _target_for(spec, metric.target)
        expected = _reference_value(reference, metric)
        valid, predicted, error = _project(answers.get(metric.target), metric, target, case)
        score: dict[str, Any] = {
            "kind": metric.kind,
            "valid": valid,
            "predicted": predicted,
            "expected": expected,
        }
        if error is not None:
            score["error"] = error

        if metric.kind == "exact":
            if valid and target.cardinality == "many":
                try:
                    prediction_set = _projected_set(predicted)
                    reference_set = _hashable_set(expected, description="Reference values")
                    score["value"] = float(prediction_set == reference_set)
                except ValueError as exc:
                    score["valid"] = False
                    score["value"] = 0.0
                    score["error"] = str(exc)
            else:
                score["value"] = float(valid and predicted == expected)
        elif metric.kind in {"binary_f1", "binary_precision"}:
            truth_positive = _binary_reference(expected, metric, reference)
            prediction_positive = _binary_prediction(predicted, metric) if valid else None
            # An invalid prediction always incurs an error, including on a
            # negative example. This prevents failures from becoming true negatives.
            score["value"] = float(valid and prediction_positive == truth_positive)
            score["truth_positive"] = truth_positive
            score["prediction_positive"] = prediction_positive
        elif metric.kind == "set_f1":
            truth = _hashable_set(
                expected, description=f"Reference {reference.case_id!r} label {metric.label!r}"
            )
            if valid:
                try:
                    prediction = _projected_set(predicted)
                except ValueError as exc:
                    valid = False
                    score["valid"] = False
                    score["error"] = str(exc)
                    prediction = set()
            else:
                prediction = set()
            overlap = len(prediction & truth) if valid else 0
            score["value"] = (
                0.0
                if not valid
                else 1.0
                if not prediction and not truth
                else 2.0 * overlap / (len(prediction) + len(truth))
            )
            score["overlap"] = overlap
            score["predicted_count"] = len(prediction)
            score["reference_count"] = len(truth)
        elif metric.kind == "mae":
            penalty = metric.failure_penalty
            if penalty is None:
                raise ValueError(f"Metric {metric.id!r} needs a failure_penalty")
            reference_number = _finite_number(expected, description="Reference value")
            if valid:
                try:
                    prediction_number = _finite_number(predicted, description="Predicted value")
                    distance = abs(prediction_number - reference_number)
                    if not math.isfinite(distance):
                        raise ValueError("Absolute error is not finite")
                except ValueError as exc:
                    valid = False
                    score["valid"] = False
                    score["error"] = str(exc)
            score["value"] = distance if valid else float(penalty)
        else:
            raise ValueError(f"Unsupported metric kind {metric.kind!r}")

        scores[metric.id] = score
    return scores


def _mean(values: Sequence[float]) -> float:
    return math.fsum(value / len(values) for value in values) if values else 0.0


def _percentile(values: Sequence[float], proportion: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _stage_summary(rows: Sequence[Mapping[str, Any]], stage_name: str) -> dict[str, Any]:
    stages = [row["stages"][stage_name] for row in rows]
    times = [float(stage["elapsed_seconds"]) for stage in stages]
    costs = [stage.get("cost_usd") for stage in stages]
    recorded = [
        float(stage.get("recorded_cost_usd", cost or 0.0))
        for stage, cost in zip(stages, costs)
    ]
    complete = [
        bool(stage.get("cost_complete")) and isinstance(cost, (int, float))
        for stage, cost in zip(stages, costs)
    ]
    recorded_seconds = sum(times)
    return {
        "total_seconds": recorded_seconds,
        "recorded_total_seconds": recorded_seconds,
        "mean_seconds": _mean(times),
        "p50_seconds": _percentile(times, 0.50),
        "p95_seconds": _percentile(times, 0.95),
        "latency_complete": True,
        "total_cost_usd": sum(float(cost) for cost in costs) if all(complete) else None,
        "recorded_cost_usd": sum(recorded),
        "cost_complete_cases": sum(complete),
    }


def _paired_delta(
    spec: EvaluationSpec,
    by_mode: Mapping[str, Mapping[str, Mapping[str, Any]]],
    mode_summaries: Mapping[str, Mapping[str, Any]],
    *,
    arm_name: str,
    baseline_name: str,
    case_ids: set[str],
) -> dict[str, Any]:
    """Compare two modes on the same cases, preserving metric direction."""

    arm = mode_summaries[arm_name]
    baseline = mode_summaries[baseline_name]
    exact_comparisons: dict[str, dict[str, int]] = {}
    for metric in spec.metrics:
        if metric.kind != "exact":
            continue
        wins = losses = ties = 0
        for case_id in sorted(case_ids):
            arm_score = by_mode[arm_name][case_id]["scores"][metric.id]["value"]
            baseline_score = by_mode[baseline_name][case_id]["scores"][metric.id]["value"]
            difference = arm_score - baseline_score
            if difference > 0:
                wins += 1
            elif difference < 0:
                losses += 1
            else:
                ties += 1
        exact_comparisons[metric.id] = {"wins": wins, "losses": losses, "ties": ties}

    delta_metrics = {
        metric.id: arm["metrics"][metric.id]["value"]
        - baseline["metrics"][metric.id]["value"]
        for metric in spec.metrics
    }
    baseline_cost = baseline["stages"]["total"]["total_cost_usd"]
    arm_cost = arm["stages"]["total"]["total_cost_usd"]
    cost_delta = (
        arm_cost - baseline_cost
        if arm_cost is not None and baseline_cost is not None
        else None
    )
    gain_per_cost: dict[str, float | None] = {}
    for metric in spec.metrics:
        if cost_delta is None or not math.isfinite(cost_delta) or cost_delta <= 0:
            gain_per_cost[metric.id] = None
            continue
        oriented_gain = delta_metrics[metric.id] * (
            -1 if metric.kind == "mae" else 1
        )
        ratio = oriented_gain / cost_delta
        gain_per_cost[metric.id] = ratio if math.isfinite(ratio) else None

    arm_mean = arm["stages"]["total"]["mean_seconds"]
    baseline_mean = baseline["stages"]["total"]["mean_seconds"]
    return {
        "metrics": delta_metrics,
        "quality_gain_per_extra_usd": gain_per_cost,
        "joint_exact_accuracy": (
            arm["joint_exact_accuracy"] - baseline["joint_exact_accuracy"]
            if arm["joint_exact_accuracy"] is not None
            and baseline["joint_exact_accuracy"] is not None
            else None
        ),
        "mean_total_seconds": (
            arm_mean - baseline_mean
            if arm_mean is not None and baseline_mean is not None
            else None
        ),
        "total_cost_usd": cost_delta,
        "cost_usd_per_case": cost_delta / len(case_ids) if cost_delta is not None else None,
        "exact_metric_case_comparisons": exact_comparisons,
    }


def summarize_runs(
    spec: EvaluationSpec,
    references: Mapping[str, ReferenceCase],
    records: Sequence[Mapping[str, Any]],
    modes: Sequence[str],
    *,
    untracked_modes: Sequence[str] = (),
) -> dict[str, Any]:
    """Aggregate paired attempts, retaining every selected case in the denominator."""

    case_ids = set(references)
    if not case_ids:
        raise ValueError("Cannot summarize an experiment with no cases")
    untracked = set(untracked_modes)
    if untracked - set(modes):
        raise ValueError(f"Unknown untracked modes: {sorted(untracked - set(modes))}")
    summary: dict[str, Any] = {
        "benchmark_id": spec.benchmark_id,
        "cases": len(case_ids),
        "modes": {},
        "untracked_cost_modes": sorted(untracked),
        "untracked_latency_modes": sorted(untracked),
    }
    by_mode: dict[str, dict[str, Mapping[str, Any]]] = {}

    for mode in modes:
        mode_rows = [row for row in records if row.get("mode") == mode]
        ids = [str(row.get("case_id")) for row in mode_rows]
        if len(ids) != len(case_ids) or set(ids) != case_ids:
            raise ValueError(f"Mode {mode!r} is missing cases or has duplicate case IDs")
        by_mode[mode] = {str(row["case_id"]): row for row in mode_rows}
        metrics: dict[str, Any] = {}
        exact_metric_ids = [metric.id for metric in spec.metrics if metric.kind == "exact"]
        for metric in spec.metrics:
            entries = [row["scores"][metric.id] for row in mode_rows]
            invalid = sum(not entry["valid"] for entry in entries)
            values = [float(entry["value"]) for entry in entries]
            metric_summary: dict[str, Any] = {
                "kind": metric.kind,
                "value": _mean(values),
                "higher_is_better": metric.kind != "mae",
                "denominator": len(case_ids),
                "invalid_answers": invalid,
                "coverage": (len(case_ids) - invalid) / len(case_ids),
            }
            if metric.kind in {"binary_f1", "binary_precision"}:
                tp = fp = fn = tn = 0
                for entry in entries:
                    truth = bool(entry["truth_positive"])
                    prediction = entry["prediction_positive"]
                    if prediction is None:
                        if truth:
                            fn += 1
                        else:
                            fp += 1
                    elif prediction and truth:
                        tp += 1
                    elif prediction and not truth:
                        fp += 1
                    elif not prediction and truth:
                        fn += 1
                    else:
                        tn += 1
                f1_denominator = 2 * tp + fp + fn
                precision_denominator = tp + fp
                if metric.kind == "binary_f1":
                    value = 2 * tp / f1_denominator if f1_denominator else 0.0
                else:
                    value = tp / precision_denominator if precision_denominator else 0.0
                metric_summary.update(
                    {
                        "value": value,
                        "accuracy": _mean(values),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                        "predicted_positive_denominator": precision_denominator,
                    }
                )
            elif metric.kind == "set_f1":
                overlap = sum(entry["overlap"] for entry in entries)
                predicted_count = sum(entry["predicted_count"] for entry in entries)
                reference_count = sum(entry["reference_count"] for entry in entries)
                denominator = predicted_count + reference_count
                metric_summary["micro_f1"] = (
                    2 * overlap / denominator if denominator else float(invalid == 0)
                )
            metrics[metric.id] = metric_summary

        status_counts = {
            status: sum(row.get("status") == status for row in mode_rows)
            for status in ("ok", "partial", "error")
        }
        stages = {
            name: _stage_summary(mode_rows, name)
            for name in ("planner", "judge", "total")
        }
        if mode in untracked:
            # An interrupted pair may have spent time and money before its
            # durable row existed. Keep observed totals, but do not present
            # them as complete latency or cost for the mode.
            for stage in stages.values():
                stage["total_seconds"] = None
                stage["mean_seconds"] = None
                stage["p50_seconds"] = None
                stage["p95_seconds"] = None
                stage["latency_complete"] = False
                stage["total_cost_usd"] = None
        summary["modes"][mode] = {
            "attempts": len(mode_rows),
            "status": status_counts,
            "completion_rate": status_counts["ok"] / len(case_ids),
            "metrics": metrics,
            "jev_calls": sum(row["jev_calls"] for row in mode_rows),
            "http_attempts": sum(row["http_attempts"] for row in mode_rows),
            "retries": sum(row["retries"] for row in mode_rows),
            "joint_exact_accuracy": (
                _mean(
                    [
                        float(
                            all(
                                row["scores"][metric_id]["value"] == 1.0
                                for metric_id in exact_metric_ids
                            )
                        )
                        for row in mode_rows
                    ]
                )
                if exact_metric_ids
                else None
            ),
            "stages": stages,
        }

    if "direct" in by_mode:
        summary["delta_vs_direct"] = {
            mode: _paired_delta(
                spec,
                by_mode,
                summary["modes"],
                arm_name=mode,
                baseline_name="direct",
                case_ids=case_ids,
            )
            for mode in modes
            if mode != "direct"
        }
    if "fixed" in by_mode and "adaptive" in by_mode:
        summary["delta_vs_fixed"] = {
            "adaptive": _paired_delta(
                spec,
                by_mode,
                summary["modes"],
                arm_name="adaptive",
                baseline_name="fixed",
                case_ids=case_ids,
            )
        }
    return summary
