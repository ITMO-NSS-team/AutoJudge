from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType

from autojudge.agent_pool import AgentPool
from autojudge.pipeline.types import PipelineTrace


@dataclass(frozen=True, slots=True)
class EvaluationTelemetry:
    """Persisted evaluation details for optimisation history."""

    quality: dict
    cost: float
    budget_spent: float
    trace: PipelineTrace
    judge_result: dict
    timestamp: datetime
    result_output: Any | None = None
    latency_seconds: float | None = None
    pipeline_tokens: Mapping[str, int] | None = None
    pipeline_cost: Mapping[str, float] | None = None
    mermaid_graph: str | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError("Evaluation cost cannot be negative")
        if self.budget_spent < 0:
            raise ValueError("Budget spent cannot be negative")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        if self.pipeline_tokens is not None:
            object.__setattr__(
                self, "pipeline_tokens", MappingProxyType(dict(self.pipeline_tokens))
            )
        if self.pipeline_cost is not None:
            object.__setattr__(
                self, "pipeline_cost", MappingProxyType(dict(self.pipeline_cost))
            )


def clone_pool(pool: AgentPool) -> AgentPool:
    """Return a deep copy of an :class:`AgentPool`."""
    return AgentPool([deepcopy(agent) for agent in pool])


@dataclass(frozen=True, slots=True)
class PoolCandidate:
    """Candidate agent pool subject to evaluation."""

    pool: AgentPool

    def clone(self) -> PoolCandidate:
        return PoolCandidate(clone_pool(self.pool))


@dataclass(frozen=True, slots=True)
class PoolEvaluationRecord:
    """Association between a pool candidate and telemetry."""

    candidate: PoolCandidate
    telemetry: EvaluationTelemetry


@dataclass(frozen=True, slots=True)
class AgentPoolNeighbour:
    """Perturbation operator used during VNS shaking."""

    name: str
    transform: Callable[[AgentPool, int, random.Random], AgentPool]

    def apply(
        self, pool: AgentPool, amplitude: int, rng: random.Random
    ) -> PoolCandidate:
        mutated = self.transform(clone_pool(pool), amplitude, rng)
        return PoolCandidate(mutated)


class AgentPoolNeighborhood:
    """Collection of neighbourhood operators for AgentPool perturbations."""

    def __init__(
        self,
        operators: Sequence[AgentPoolNeighbour],
        *,
        rng: random.Random | None = None,
    ) -> None:
        if not operators:
            raise ValueError("At least one neighbourhood operator is required")
        self._operators = tuple(operators)
        self._rng = rng or random.Random()

    def neighbours(self, pool: AgentPool, amplitude: int) -> tuple[PoolCandidate, ...]:
        if amplitude <= 0:
            raise ValueError("Amplitude must be positive")
        return tuple(
            operator.apply(pool, amplitude, self._rng) for operator in self._operators
        )
