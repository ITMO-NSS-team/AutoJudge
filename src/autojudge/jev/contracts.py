"""Versioned input, rubric, and planning contracts for Jev evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NONE_OPTION_ID = "none"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EvidenceEvent(_StrictModel):
    """One event in the normalized order; source_step keeps the original index."""

    id: str = Field(min_length=1)
    index: int = Field(ge=1)
    actor: str | None = None
    role: str | None = None
    content: Any
    source_step: int | None = Field(default=None, ge=0)
    source_id: str | None = None


class EvaluationCase(_StrictModel):
    """Evidence visible to the judge. Reference labels are stored separately."""

    version: Literal[1] = 1
    case_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    events: list[EvidenceEvent] = Field(min_length=1)
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_events(self) -> EvaluationCase:
        seen: set[str] = set()
        for expected_index, event in enumerate(self.events, start=1):
            if event.id in seen:
                raise ValueError(f"Duplicate event id: {event.id}")
            seen.add(event.id)
            if event.index != expected_index:
                raise ValueError(
                    f"Event {event.id} has index {event.index}; expected {expected_index}"
                )
        return self


class Option(_StrictModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class TargetSpec(_StrictModel):
    """A result field and the source of its model-generated alternatives."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    source: Literal["events", "actors", "options"]
    cardinality: Literal["one", "many"] = "one"
    options: list[Option] = Field(default_factory=list)
    allow_none: bool = False
    none_label: str = Field(default="None of the above", min_length=1)

    @model_validator(mode="after")
    def validate_options(self) -> TargetSpec:
        option_ids = [option.id for option in self.options]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError(f"Target {self.id} has duplicate option ids")
        if self.source == "options" and not self.options:
            raise ValueError(f"Target {self.id} requires static options")
        if self.source != "options" and self.options:
            raise ValueError(f"Target {self.id} derives options from {self.source}")
        if self.allow_none and NONE_OPTION_ID in option_ids:
            raise ValueError(f"Target {self.id} reserves option id {NONE_OPTION_ID}")
        return self


class MetricSpec(_StrictModel):
    """How a target is projected and compared with a separate reference label."""

    id: str = Field(min_length=1)
    target: str = Field(min_length=1)
    kind: Literal["exact", "binary_f1", "binary_precision", "set_f1", "mae"] = "exact"
    projection: Literal["id", "actor", "index", "source_step"] = "id"
    label: str = Field(min_length=1)
    positive_id: str | None = None
    failure_penalty: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_kind_fields(self) -> MetricSpec:
        if self.kind in {"binary_f1", "binary_precision"}:
            if not self.positive_id:
                raise ValueError(f"Metric {self.id} requires positive_id for {self.kind}")
        elif self.positive_id is not None:
            raise ValueError(f"Metric {self.id} uses positive_id only with binary metrics")
        if self.kind == "mae":
            if self.failure_penalty is None:
                raise ValueError(f"Metric {self.id} requires failure_penalty for mae")
        elif self.failure_penalty is not None:
            raise ValueError(f"Metric {self.id} uses failure_penalty only with mae")
        return self


class EvaluationSpec(_StrictModel):
    """Benchmark-specific rubric and scoring, independent of source data format."""

    version: Literal[1] = 1
    benchmark_id: str = Field(min_length=1)
    taxonomy: str
    targets: list[TargetSpec] = Field(min_length=1)
    metrics: list[MetricSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> EvaluationSpec:
        targets_by_id = {target.id: target for target in self.targets}
        if len(targets_by_id) != len(self.targets):
            raise ValueError("Target ids must be unique")
        metric_ids = {metric.id for metric in self.metrics}
        if len(metric_ids) != len(self.metrics):
            raise ValueError("Metric ids must be unique")
        for metric in self.metrics:
            target = targets_by_id.get(metric.target)
            if target is None:
                raise ValueError(f"Metric {metric.id} references unknown target {metric.target}")
            if metric.projection in {"actor", "index", "source_step"} and target.source != "events":
                raise ValueError(
                    f"Metric {metric.id} projects event fields from non-event target {target.id}"
                )
            if metric.kind in {"binary_f1", "binary_precision"}:
                if target.source != "options" or target.cardinality != "one":
                    raise ValueError(f"Metric {metric.id} needs one static option target")
                valid_ids = {option.id for option in target.options}
                if target.allow_none:
                    valid_ids.add(NONE_OPTION_ID)
                if metric.positive_id not in valid_ids:
                    raise ValueError(f"Metric {metric.id} has unknown positive_id")
                if metric.projection != "id":
                    raise ValueError(f"Metric {metric.id} needs id projection")
            if metric.kind == "set_f1" and target.cardinality != "many":
                raise ValueError(f"Metric {metric.id} needs a many-cardinality target")
            if metric.kind == "mae" and target.cardinality != "one":
                raise ValueError(f"Metric {metric.id} needs a one-cardinality target")
        unscored = set(targets_by_id) - {metric.target for metric in self.metrics}
        if unscored:
            raise ValueError(
                f"Every evaluation target needs a metric; unscored targets: {sorted(unscored)}"
            )
        return self


class ReferenceCase(_StrictModel):
    """Gold labels kept out of judge and planner input."""

    case_id: str = Field(min_length=1)
    labels: dict[str, Any] = Field(min_length=1)


class SpecialistPlan(_StrictModel):
    name: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    target_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=1500)


class JudgePlan(_StrictModel):
    specialists: list[SpecialistPlan] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_names(self) -> JudgePlan:
        names = [specialist.name for specialist in self.specialists]
        if len(names) != len(set(names)) or "FINAL_AGGREGATOR" in names:
            raise ValueError("Specialist names must be unique and exclude FINAL_AGGREGATOR")
        return self
