"""Reject normalized cases that cannot be scored before calling a model.

Gold labels remain outside the judge input. This module checks label domains
and whether categorical references for one target are jointly satisfiable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .contracts import (
    NONE_OPTION_ID,
    EvaluationCase,
    EvaluationSpec,
    MetricSpec,
    ReferenceCase,
    TargetSpec,
)
from .engine import Candidate, candidates_for


def _finite_numbers(value: Any, location: str, active: set[int]) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    if not isinstance(value, (Mapping, list, tuple)):
        return
    identity = id(value)
    if identity in active:
        raise ValueError(f"{location} contains a cyclic value")
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            for key, item in value.items():
                _finite_numbers(item, f"{location}[{key!r}]", active)
        else:
            for index, item in enumerate(value):
                _finite_numbers(item, f"{location}[{index}]", active)
    finally:
        active.remove(identity)


def _value_key(value: Any, *, location: str) -> tuple[str, str | int | float]:
    """Use scoring-compatible scalar equality without bool/int collisions."""
    if isinstance(value, bool):
        raise ValueError(f"{location} must not be a boolean")
    if isinstance(value, str):
        return "string", value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{location} must be finite")
        return "number", value
    raise ValueError(f"{location} must be a string or finite number")


def _value_set(values: list[Any], *, location: str) -> set[tuple[str, str | int | float]]:
    keys = {_value_key(value, location=location) for value in values}
    if len(keys) != len(values):
        raise ValueError(f"{location} contains duplicate projected values")
    return keys


def _finite_number(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{location} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{location} must be a finite number")
    return number


def _project_candidates(
    candidates: list[Candidate], metric: MetricSpec, case_id: str
) -> list[Any]:
    values: list[Any] = []
    for candidate in candidates:
        value = getattr(candidate, metric.projection)
        if value is None or (metric.projection == "actor" and not value):
            raise ValueError(
                f"Case {case_id!r}, metric {metric.id!r}: candidate "
                f"{candidate.id!r} lacks required {metric.projection!r}"
            )
        values.append(value)
    return values


def _binary_truth(
    value: Any, metric: MetricSpec, candidate_ids: set[str], allow_none: bool
) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if value is None:
        if not allow_none:
            raise ValueError("gold no-match requires allow_none=true")
        return metric.positive_id == NONE_OPTION_ID
    if isinstance(value, str):
        if value in candidate_ids:
            # A literal option ID "none" is distinct from a no-match answer.
            return value == metric.positive_id
        if value == NONE_OPTION_ID:
            if not allow_none:
                raise ValueError("gold no-match requires allow_none=true")
            return metric.positive_id == NONE_OPTION_ID
        raise ValueError(f"gold option {value!r} is not a candidate")
    raise ValueError("binary gold must be a boolean, 0/1, option ID, or null")


def _validate_metric(
    case: EvaluationCase,
    metric: MetricSpec,
    candidates: list[Candidate],
    target_cardinality: str,
    allow_none: bool,
    gold: Any,
) -> None:
    location = f"Case {case.case_id!r}, metric {metric.id!r}"
    values = _project_candidates(candidates, metric, case.case_id)

    if metric.kind in {"binary_f1", "binary_precision"}:
        if target_cardinality != "one" or metric.projection != "id":
            raise ValueError(f"{location}: binary metrics need one ID selection")
        candidate_ids = {candidate.id for candidate in candidates}
        try:
            truth = _binary_truth(gold, metric, candidate_ids, allow_none)
        except ValueError as exc:
            raise ValueError(f"{location}: {exc}") from exc
        positive = metric.positive_id
        if truth:
            possible = positive in candidate_ids or (allow_none and positive == NONE_OPTION_ID)
        else:
            possible = any(candidate_id != positive for candidate_id in candidate_ids) or (
                allow_none and positive != NONE_OPTION_ID
            )
        if not possible:
            raise ValueError(f"{location}: gold binary class is not reachable")
        return

    if metric.kind == "mae":
        if target_cardinality != "one":
            raise ValueError(f"{location}: mae needs one selection")
        for candidate, value in zip(candidates, values):
            _finite_number(value, location=f"{location} candidate {candidate.id!r}")
        _finite_number(gold, location=f"{location} gold")
        return

    if metric.kind not in {"exact", "set_f1"}:
        raise ValueError(f"{location}: unsupported metric kind {metric.kind!r}")

    if target_cardinality == "one":
        if metric.kind != "exact":
            raise ValueError(f"{location}: set_f1 needs many-cardinality target")
        if gold is None:
            if not allow_none:
                raise ValueError(f"{location}: gold no-match requires allow_none=true")
            return
        gold_key = _value_key(gold, location=f"{location} gold")
        domain = {_value_key(value, location=f"{location} candidate") for value in values}
        if gold_key not in domain:
            suffix = "; use JSON null for no-match" if gold == NONE_OPTION_ID else ""
            raise ValueError(f"{location}: gold {gold!r} is not a candidate{suffix}")
        return

    if target_cardinality != "many":
        raise ValueError(f"{location}: unknown target cardinality {target_cardinality!r}")
    if not isinstance(gold, list):
        raise ValueError(f"{location}: many-cardinality gold must be a JSON array")
    if not gold and not allow_none:
        raise ValueError(f"{location}: empty gold set requires allow_none=true")
    gold_set = _value_set(gold, location=f"{location} gold set")
    domain = {_value_key(value, location=f"{location} candidate") for value in values}
    missing = gold_set - domain
    if missing:
        raise ValueError(f"{location}: gold contains values outside the candidate domain")


def _validate_joint_metrics(
    case: EvaluationCase,
    target: TargetSpec,
    candidates: list[Candidate],
    metrics: list[MetricSpec],
    labels: Mapping[str, Any],
) -> None:
    """Check that one target answer can satisfy all its categorical labels."""
    joint_metrics = [metric for metric in metrics if metric.kind != "mae"]
    if len(joint_metrics) < 2:
        return

    location = f"Case {case.case_id!r}, target {target.id!r}"
    if target.cardinality == "one":
        candidate_ids = {candidate.id for candidate in candidates}
        exact = [
            (
                metric,
                None
                if labels[metric.label] is None
                else _value_key(
                    labels[metric.label], location=f"{location} metric {metric.id!r}"
                ),
            )
            for metric in joint_metrics
            if metric.kind == "exact"
        ]
        binary = [
            (
                metric,
                _binary_truth(labels[metric.label], metric, candidate_ids, target.allow_none),
            )
            for metric in joint_metrics
            if metric.kind in {"binary_f1", "binary_precision"}
        ]
        for candidate in candidates:
            matches_exact = all(
                expected is not None
                and _value_key(getattr(candidate, metric.projection), location=location) == expected
                for metric, expected in exact
            )
            matches_binary = all(
                (candidate.id == metric.positive_id) == truth for metric, truth in binary
            )
            if matches_exact and matches_binary:
                return
        none_matches_exact = all(expected is None for _, expected in exact)
        none_matches_binary = all(
            (NONE_OPTION_ID == metric.positive_id) == truth for metric, truth in binary
        )
        if target.allow_none and none_matches_exact and none_matches_binary:
            return
    else:
        expected_sets = [
            (
                metric,
                _value_set(labels[metric.label], location=f"{location} metric {metric.id!r}"),
            )
            for metric in joint_metrics
        ]
        compatible = [
            candidate
            for candidate in candidates
            if all(
                _value_key(getattr(candidate, metric.projection), location=location) in expected
                for metric, expected in expected_sets
            )
        ]
        if all(
            expected
            <= {
                _value_key(getattr(candidate, metric.projection), location=location)
                for candidate in compatible
            }
            for metric, expected in expected_sets
        ):
            return

    metric_ids = [metric.id for metric in joint_metrics]
    raise ValueError(f"{location}: no selection satisfies metrics {metric_ids!r} together")


def validate_case_against_spec(
    case: EvaluationCase, spec: EvaluationSpec, reference: ReferenceCase
) -> None:
    """Raise ``ValueError`` if one normalized case is impossible to evaluate.

    Call this before inference. References are used only for validation and
    must never be passed to the planner or Jev evaluator.
    """
    if case.case_id != reference.case_id:
        raise ValueError(
            f"Case ID {case.case_id!r} does not match reference ID {reference.case_id!r}"
        )

    active: set[int] = set()
    for event in case.events:
        _finite_numbers(event.content, f"Case {case.case_id!r}, event {event.id!r} content", active)
    _finite_numbers(case.metadata, f"Case {case.case_id!r} metadata", active)
    _finite_numbers(reference.labels, f"Case {case.case_id!r} reference labels", active)

    targets = {target.id: target for target in spec.targets}
    candidate_cache: dict[str, list[Candidate]] = {}
    for target in spec.targets:
        try:
            candidates = candidates_for(case, target)
        except ValueError as exc:
            raise ValueError(
                f"Case {case.case_id!r}, target {target.id!r}: {exc}"
            ) from exc
        candidate_cache[target.id] = candidates
        if not candidates and not target.allow_none:
            raise ValueError(
                f"Case {case.case_id!r}, target {target.id!r}: no selectable candidates"
            )

    metrics_by_target: dict[str, list[MetricSpec]] = {target.id: [] for target in spec.targets}
    for metric in spec.metrics:
        target = targets[metric.target]
        candidates = candidate_cache[target.id]
        metrics_by_target[target.id].append(metric)
        if metric.label not in reference.labels:
            raise ValueError(
                f"Case {case.case_id!r}, metric {metric.id!r}: "
                f"missing gold label {metric.label!r}"
            )
        _validate_metric(
            case,
            metric,
            candidates,
            target.cardinality,
            target.allow_none,
            reference.labels[metric.label],
        )

    for target in spec.targets:
        _validate_joint_metrics(
            case,
            target,
            candidate_cache[target.id],
            metrics_by_target[target.id],
            reference.labels,
        )
